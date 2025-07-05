import React, { useState } from 'react';
import CardInput from '../components/CardInput';
import GameStateInput from '../components/GameStateInput';
import StrategyDisplay from '../components/StrategyDisplay';
import strategyData from '../../../backend/bot/analysis/blueprint_trainer_extended.json';

function StrategyLookup() {
    const [gameState, setGameState] = useState({
        holeCards: [],
        communityCards: [],
        actions: [], // Array of {action, amount} objects
        initialPotSize: 3, // Starting pot
        potSize: 3, // Calculated from actions
        lastBetAmount: 0, // Calculated from actions
        currentBet: 0, // Calculated from actions
        street: 'preflop'
    });

    const [foundStrategy, setFoundStrategy] = useState(null);
    const [infoSetKey, setInfoSetKey] = useState('');
    const [debugInfo, setDebugInfo] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const findStrategy = async () => {
        try {
            setLoading(true);
            setError(null);

            const actionsForAPI = gameState.actions.map(actionObj => ({
                action: actionObj.action,
                amount: actionObj.amount
            }));

            const gameStateForAPI = {
                potSize: gameState.potSize,
                playerStack: 100, // Default
                currentBet: gameState.currentBet,
                playerContribution: 0, // Simplified
                bigBlind: 2
            };

            const response = await fetch('http://localhost:5000/api/evaluate-hand', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    holeCards: gameState.holeCards,
                    communityCards: gameState.communityCards,
                    actions: actionsForAPI,
                    gameState: gameStateForAPI
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            setInfoSetKey(result.infoSetKey);
            setDebugInfo({
                cardBucket: result.cardBucket,
                strengthBucket: result.strengthBucket,
                actionPattern: result.actionPattern,
                street: getStreetFromCommunityCards(gameState.communityCards)
            });

            // Look up strategy using the correct key from Python
            const strategy = strategyData.normalized_strategies[result.infoSetKey];
            setFoundStrategy(strategy || null);

        } catch (error) {
            console.error('API Error:', error);
            setError(error.message);
            setFoundStrategy(null);
        } finally {
            setLoading(false);
        }
    };

    const getStreetFromCommunityCards = (communityCards) => {
        if (!communityCards || communityCards.length === 0) return 'preflop';
        if (communityCards.length === 3) return 'flop';
        if (communityCards.length === 4) return 'turn';
        if (communityCards.length === 5) return 'river';
        return 'preflop';
    };


    return (
        <div className="app-container">
            <h1 style={{ color: 'white' }}>Strategy Lookup</h1>

            <div className="strategy-lookup-container">
                <CardInput gameState={gameState} setGameState={setGameState} />
                <GameStateInput gameState={gameState} setGameState={setGameState} />

                <button
                    onClick={findStrategy}
                    className="lookup-button"
                    disabled={loading}
                    style={{
                        opacity: loading ? 0.7 : 1,
                        cursor: loading ? 'not-allowed' : 'pointer'
                    }}
                >
                    {loading ? 'Analyzing...' : 'Find Strategy'}
                </button>

                {error && (
                    <div style={{
                        color: '#ff6b6b',
                        margin: '10px 0',
                        padding: '10px',
                        backgroundColor: '#330',
                        borderRadius: '5px',
                        border: '1px solid #ff6b6b'
                    }}>
                        <strong>Error:</strong> {error}
                        <br />
                        <small>Make sure the backend server is running on http://localhost:5000</small>
                    </div>
                )}

                {infoSetKey && (
                    <div style={{ color: 'white', margin: '10px 0', padding: '10px', backgroundColor: '#333', borderRadius: '5px' }}>
                        <h3>Debug Information: <span style={{ fontSize: '12px', color: '#4CAF50' }}>🐍 From Python API</span></h3>
                        <p><strong>Info Set Key:</strong> <code>{infoSetKey}</code></p>
                        <p><strong>Card Bucket:</strong> {debugInfo.cardBucket}</p>
                        {debugInfo.strengthBucket && <p><strong>Strength Bucket:</strong> {debugInfo.strengthBucket}</p>}
                        <p><strong>Action Pattern:</strong> '{debugInfo.actionPattern}'</p>
                        <p><strong>Street:</strong> {debugInfo.street}</p>
                    </div>
                )}

                <StrategyDisplay strategy={foundStrategy} />
            </div>
        </div>
    );
}

export default StrategyLookup;

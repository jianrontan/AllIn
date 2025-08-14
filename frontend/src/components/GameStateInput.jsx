// C:\Ron\AllIn\frontend\src\components\GameStateInput.jsx
import React, { useState, useEffect } from 'react';

function GameStateInput({ gameState, setGameState }) {
    const [pendingBetAmount, setPendingBetAmount] = useState(0);
    const [showBetInput, setShowBetInput] = useState(false);
    const [pendingAction, setPendingAction] = useState(null);
    const [legalActions, setLegalActions] = useState([]);

    // Fetch legal actions when game state changes
    useEffect(() => {
        const fetchLegalActions = async () => {
            try {
                const response = await fetch('http://localhost:5000/api/get-legal-actions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        actions: gameState.actions,
                        gameState: gameState,
                        communityCards: gameState.communityCards
                    })
                });

                const data = await response.json();
                if (data.legalActions) {
                    setLegalActions(data.legalActions);
                }
            } catch (error) {
                console.error('Error fetching legal actions:', error);
                setLegalActions(['check', 'bet', 'call', 'raise', 'fold']); // Fallback
            }
        };

        fetchLegalActions();
    }, [gameState.actions, gameState.communityCards]);

    // Calculate current game state from action history
    const calculateGameStateFromActions = (actions, initialPot) => {
        let currentPot = initialPot;
        let lastBetAmount = 0;
        let currentBet = 0;

        for (const action of actions) {
            if (action.action === 'bet' || action.action === 'raise') {
                currentPot += action.amount;
                lastBetAmount = action.amount;
                currentBet = action.amount;
            } else if (action.action === 'call') {
                currentPot += lastBetAmount;
            }
        }

        return { currentPot, lastBetAmount, currentBet };
    };

    const getBettingSizeOptions = (street, actionHistory, potSize) => {
        if (street === 'preflop') {
            const betCount = actionHistory.filter(a =>
                a.action === 'bet' || a.action === 'raise'
            ).length;

            if (betCount === 0) {
                return [
                    { label: '3BB', amount: 6 },
                    { label: '5BB', amount: 10 },
                    { label: '7BB', amount: 14 }
                ];
            } else if (betCount === 1) {
                return [
                    { label: '6BB', amount: 12 },
                    { label: '10BB', amount: 20 },
                    { label: '14BB', amount: 28 }
                ];
            } else {
                return [
                    { label: `0.66x pot (${(0.66 * potSize).toFixed(2)})`, amount: 0.66 * potSize },
                    { label: `1.33x pot (${(1.33 * potSize).toFixed(2)})`, amount: 1.33 * potSize },
                    { label: `2.0x pot (${(2.0 * potSize).toFixed(2)})`, amount: 2.0 * potSize }
                ];
            }
        } else {
            return [
                { label: `0.33x pot (${(0.33 * potSize).toFixed(2)})`, amount: 0.33 * potSize },
                { label: `0.66x pot (${(0.66 * potSize).toFixed(2)})`, amount: 0.66 * potSize },
                { label: `1.0x pot (${(1.0 * potSize).toFixed(2)})`, amount: 1.0 * potSize }
            ];
        }
    };

    const getCallAmount = () => {
        // Handle preflop with no actions (SB's first decision)
        if (gameState.street === 'preflop' && gameState.actions.length === 0) {
            return 1; // SB needs $1 more to match BB's $2
        }

        // Check if there's actually a bet to call
        const lastBetAction = gameState.actions
            .slice()
            .reverse()
            .find(action => action.action === 'bet' || action.action === 'raise');

        if (!lastBetAction) {
            return 0; // No bet to call
        }

        // Find the player's last contribution
        const currentPlayer = gameState.actions.length % 2;
        let playerContribution = 0;

        // For preflop, start with blind amounts
        if (gameState.street === 'preflop') {
            playerContribution = currentPlayer === 0 ? 1 : 2; // SB=1, BB=2
        }

        // Update with any bets/raises this player made
        for (let i = 0; i < gameState.actions.length; i++) {
            const action = gameState.actions[i];
            const actionPlayer = i % 2;

            if (actionPlayer === currentPlayer &&
                (action.action === 'bet' || action.action === 'raise')) {
                playerContribution = action.amount;
            }
        }

        return Math.max(0, lastBetAction.amount - playerContribution);
    };

    const isActionLegal = (actionType) => {
        if (actionType === 'call') {
            // Call is only legal if there's actually a bet to call AND it's in legal actions
            const callAmount = getCallAmount();
            return legalActions.includes('call') && callAmount > 0;
        }

        return legalActions.includes(actionType);
    };

    const getSizeNameFromAmount = (amount, street, potSize) => {
        if (street === 'preflop') {
            const betCount = gameState.actions.filter(a =>
                a.action === 'bet' || a.action === 'raise'
            ).length;

            if (betCount === 0) {
                if (amount === 6) return 'small';
                if (amount === 10) return 'medium';
                if (amount === 14) return 'large';
            } else if (betCount === 1) {
                if (amount === 12) return 'small';
                if (amount === 20) return 'medium';
                if (amount === 28) return 'large';
            } else {
                const ratio = amount / potSize;
                if (Math.abs(ratio - 0.66) < Math.abs(ratio - 1.33) && Math.abs(ratio - 0.66) < Math.abs(ratio - 2.0)) return 'small';
                if (Math.abs(ratio - 1.33) < Math.abs(ratio - 2.0)) return 'medium';
                return 'large';
            }
        } else {
            // Use your actual BET_MULTIPLIERS: 0.33, 0.66, 1.0
            const ratio = amount / potSize;
            if (Math.abs(ratio - 0.33) < Math.abs(ratio - 0.66) && Math.abs(ratio - 0.33) < Math.abs(ratio - 1.0)) return 'small';
            if (Math.abs(ratio - 0.66) < Math.abs(ratio - 1.0)) return 'medium';
            return 'large';
        }
    };

    // Update calculated values when actions change
    useEffect(() => {
        const calculated = calculateGameStateFromActions(gameState.actions, gameState.initialPotSize || 3);
        setGameState(prev => ({
            ...prev,
            potSize: calculated.currentPot,
            lastBetAmount: calculated.lastBetAmount,
            currentBet: calculated.currentBet
        }));
    }, [gameState.actions, gameState.initialPotSize]);

    const addSimpleAction = (action) => {
        // Validate action before adding
        if (!isActionLegal(action)) {
            console.warn(`Attempted illegal action: ${action}. Legal actions: ${legalActions}`);
            return;
        }

        // Special handling for call to ensure proper amount
        let amount = 0;
        if (action === 'call') {
            amount = getCallAmount();
            if (amount <= 0) {
                console.warn('Cannot call with amount <= 0');
                return;
            }
        }

        const newAction = { action, amount };
        setGameState(prev => ({
            ...prev,
            actions: [...prev.actions, newAction]
        }));
    };

    const startBetAction = (actionType) => {
        setPendingAction(actionType);
        setShowBetInput(true);
        setPendingBetAmount(0);
    };

    const cancelBetAction = () => {
        setShowBetInput(false);
        setPendingAction(null);
        setPendingBetAmount(0);
    };

    const clearAllActions = () => {
        setGameState(prev => ({
            ...prev,
            actions: [],
            potSize: prev.initialPotSize || 3,
            lastBetAmount: 0,
            currentBet: 0
        }));
    };

    const undoLastAction = () => {
        if (gameState.actions.length > 0) {
            setGameState(prev => ({
                ...prev,
                actions: prev.actions.slice(0, -1)
            }));
        }
    };

    const calculated = calculateGameStateFromActions(gameState.actions, gameState.initialPotSize || 3);

    return (
        <div style={{ color: 'white', margin: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
            <h3>Game State Builder</h3>

            {/* Initial Pot Size */}
            <div style={{ marginBottom: '20px' }}>
                <label>Initial Pot Size (before actions):</label>
                <input
                    type="number"
                    value={gameState.initialPotSize || 3}
                    onChange={(e) => setGameState(prev => ({
                        ...prev,
                        initialPotSize: parseInt(e.target.value) || 3
                    }))}
                    style={{ width: '80px', padding: '5px', marginLeft: '10px' }}
                />
            </div>

            {/* Current State Display */}
            <div style={{ marginBottom: '20px', padding: '10px', backgroundColor: '#333', borderRadius: '5px' }}>
                <strong>Current State:</strong>
                <div>Current Pot: ${calculated.currentPot}</div>
                {calculated.lastBetAmount > 0 && <div>Last Bet: ${calculated.lastBetAmount}</div>}
                <div>Action History: {gameState.actions.length === 0 ? 'None' :
                    gameState.actions.map(a => a.action + (a.amount > 0 ? `($${a.amount})` : '')).join(' → ')
                }</div>
            </div>

            {/* Bet Amount Input (when needed) */}
            {showBetInput && (
                <div style={{ margin: '10px 0', padding: '15px', border: '2px solid #ccc', borderRadius: '8px' }}>
                    <h4>Select {pendingAction} size:</h4>
                    <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '10px' }}>
                        {getBettingSizeOptions(gameState.street, gameState.actions, potBeforeActions)
                            .map((option, index) => {
                                // Determine the size category directly
                                let sizeCategory;
                                if (gameState.street === 'preflop') {
                                    const betCount = gameState.actions.filter(a => a.action === 'bet' || a.action === 'raise').length;
                                    if (betCount === 0) {
                                        sizeCategory = ['small', 'medium', 'large'][index]; // 3BB, 5BB, 7BB
                                    } else if (betCount === 1) {
                                        sizeCategory = ['small', 'medium', 'large'][index]; // 6BB, 10BB, 14BB
                                    } else {
                                        sizeCategory = ['small', 'medium', 'large'][index]; // 0.66x, 1.33x, 2.0x
                                    }
                                } else {
                                    sizeCategory = ['small', 'medium', 'large'][index]; // 0.33x, 0.66x, 1.0x
                                }

                                return (
                                    <button
                                        key={index}
                                        onClick={() => {
                                            const newAction = {
                                                action: pendingAction,
                                                amount: option.amount,
                                                size: sizeCategory  // ADD THIS - pass the size directly
                                            };
                                            setGameState(prev => ({ ...prev, actions: [...prev.actions, newAction] }));
                                            setShowBetInput(false);
                                            setPendingAction(null);
                                            setPendingBetAmount(0);
                                        }}
                                        style={{
                                            padding: '12px 20px',
                                            backgroundColor: '#4CAF50',
                                            color: 'white',
                                            border: '1px solid #45a049',
                                            borderRadius: '6px',
                                            cursor: 'pointer',
                                            fontSize: '14px',
                                            fontWeight: 'bold'
                                        }}
                                    >
                                        {option.label}
                                    </button>
                                );
                            })}
                    </div>

                    <button
                        onClick={cancelBetAction}
                        style={{
                            padding: '8px 16px',
                            backgroundColor: '#f44336',
                            color: 'white',
                            border: 'none',
                            borderRadius: '4px',
                            cursor: 'pointer'
                        }}
                    >
                        Cancel
                    </button>
                </div>
            )}

            {/* Action Buttons */}
            {!showBetInput && (
                <div style={{ marginBottom: '20px' }}>
                    <label>Available Actions:</label>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '10px' }}>
                        {legalActions.includes('check') && (
                            <button
                                onClick={() => addSimpleAction('check')}
                                style={{
                                    padding: '10px 15px',
                                    backgroundColor: '#4CAF50',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Check
                            </button>
                        )}

                        {/* REMOVED: Call and Fold buttons - these are terminal actions */}

                        {legalActions.some(action => action.startsWith('bet_')) && (
                            <button
                                onClick={() => startBetAction('bet')}
                                style={{
                                    padding: '10px 15px',
                                    backgroundColor: '#FF9800',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Bet...
                            </button>
                        )}

                        {legalActions.some(action => action.startsWith('raise_')) && (
                            <button
                                onClick={() => startBetAction('raise')}
                                style={{
                                    padding: '10px 15px',
                                    backgroundColor: '#9C27B0',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Raise...
                            </button>
                        )}
                    </div>

                    {/* Update the message for when no strategic actions are available */}
                    {!legalActions.some(action =>
                        action === 'check' ||
                        action.startsWith('bet_') ||
                        action.startsWith('raise_')
                    ) && (
                            <div style={{
                                padding: '10px',
                                backgroundColor: '#555',
                                borderRadius: '5px',
                                textAlign: 'center',
                                color: '#ccc'
                            }}>
                                No strategic actions available - only terminal actions (fold/call) remain
                            </div>
                        )}
                </div>
            )}

            {/* Control Buttons */}
            {!showBetInput && (
                <div style={{ display: 'flex', gap: '10px' }}>
                    {gameState.actions.length > 0 && (
                        <button
                            onClick={undoLastAction}
                            style={{
                                padding: '8px 12px',
                                backgroundColor: '#666',
                                color: 'white',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: 'pointer'
                            }}
                        >
                            Undo Last
                        </button>
                    )}

                    {gameState.actions.length > 0 && (
                        <button
                            onClick={clearAllActions}
                            style={{
                                padding: '8px 12px',
                                backgroundColor: '#f44336',
                                color: 'white',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: 'pointer'
                            }}
                        >
                            Clear All
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}

export default GameStateInput;

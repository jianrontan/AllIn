import React, { useState, useEffect } from 'react';

function GameStateInput({ gameState, setGameState }) {
    const [pendingBetAmount, setPendingBetAmount] = useState(0);
    const [showBetInput, setShowBetInput] = useState(false);
    const [pendingAction, setPendingAction] = useState(null); // 'bet' or 'raise'

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

    // Determine what actions are currently legal
    const getLegalActions = () => {
        const actions = gameState.actions;

        if (actions.length === 0) {
            return ['check', 'bet'];
        }

        const lastAction = actions[actions.length - 1];

        if (lastAction.action === 'check') {
            return ['check', 'bet'];
        } else if (lastAction.action === 'bet' || lastAction.action === 'raise') {
            return ['fold', 'call', 'raise'];
        } else if (lastAction.action === 'call' || lastAction.action === 'fold') {
            return []; // Round complete
        }

        return ['check', 'bet'];
    };

    const addSimpleAction = (action) => {
        const newAction = { action, amount: 0 };
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

    const confirmBetAction = () => {
        if (pendingBetAmount > 0) {
            const newAction = {
                action: pendingAction,
                amount: pendingBetAmount
            };
            setGameState(prev => ({
                ...prev,
                actions: [...prev.actions, newAction]
            }));
        }

        // Reset bet input state
        setShowBetInput(false);
        setPendingAction(null);
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

    const legalActions = getLegalActions();
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
                <div style={{
                    marginBottom: '20px',
                    padding: '15px',
                    backgroundColor: '#444',
                    borderRadius: '5px',
                    border: '2px solid #FFA500'
                }}>
                    <h4>Enter {pendingAction} amount:</h4>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span>$</span>
                        <input
                            type="number"
                            value={pendingBetAmount}
                            onChange={(e) => setPendingBetAmount(parseInt(e.target.value) || 0)}
                            style={{ width: '100px', padding: '8px', fontSize: '16px' }}
                            autoFocus
                        />
                        <button
                            onClick={confirmBetAction}
                            disabled={pendingBetAmount <= 0}
                            style={{
                                padding: '8px 15px',
                                backgroundColor: pendingBetAmount > 0 ? '#4CAF50' : '#666',
                                color: 'white',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: pendingBetAmount > 0 ? 'pointer' : 'not-allowed'
                            }}
                        >
                            Confirm {pendingAction} ${pendingBetAmount}
                        </button>
                        <button
                            onClick={cancelBetAction}
                            style={{
                                padding: '8px 15px',
                                backgroundColor: '#f44336',
                                color: 'white',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: 'pointer'
                            }}
                        >
                            Cancel
                        </button>
                    </div>

                    {/* Quick bet suggestions */}
                    <div style={{ marginTop: '10px' }}>
                        <span style={{ fontSize: '12px' }}>Quick amounts: </span>
                        {[
                            Math.round(calculated.currentPot * 0.33), // 33% pot
                            Math.round(calculated.currentPot * 0.66), // 66% pot  
                            calculated.currentPot, // pot bet
                            Math.round(calculated.currentPot * 1.5) // 150% pot
                        ].map(amount => (
                            <button
                                key={amount}
                                onClick={() => setPendingBetAmount(amount)}
                                style={{
                                    margin: '0 5px',
                                    padding: '4px 8px',
                                    backgroundColor: '#2196F3',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '3px',
                                    cursor: 'pointer',
                                    fontSize: '12px'
                                }}
                            >
                                ${amount}
                            </button>
                        ))}
                    </div>
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

                        {legalActions.includes('call') && (
                            <button
                                onClick={() => addSimpleAction('call')}
                                style={{
                                    padding: '10px 15px',
                                    backgroundColor: '#2196F3',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Call ${calculated.lastBetAmount}
                            </button>
                        )}

                        {legalActions.includes('fold') && (
                            <button
                                onClick={() => addSimpleAction('fold')}
                                style={{
                                    padding: '10px 15px',
                                    backgroundColor: '#f44336',
                                    color: 'white',
                                    border: 'none',
                                    borderRadius: '5px',
                                    cursor: 'pointer'
                                }}
                            >
                                Fold
                            </button>
                        )}

                        {legalActions.includes('bet') && (
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

                        {legalActions.includes('raise') && (
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

                    {legalActions.length === 0 && (
                        <div style={{
                            padding: '10px',
                            backgroundColor: '#555',
                            borderRadius: '5px',
                            textAlign: 'center',
                            color: '#ccc'
                        }}>
                            Betting round complete
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

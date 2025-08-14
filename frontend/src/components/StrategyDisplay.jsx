// C:\Ron\AllIn\frontend\src\components\StrategyDisplay.jsx
import React, { useState } from 'react';

function StrategyDisplay({ strategy }) {
    const [selectedAction, setSelectedAction] = useState(null);

    const getRandomAction = () => {
        if (!strategy || !strategy.average_strategy) return null;

        const actions = Object.keys(strategy.average_strategy);
        const probabilities = Object.values(strategy.average_strategy);

        const random = Math.random();
        let cumulativeProbability = 0;

        for (let i = 0; i < actions.length; i++) {
            cumulativeProbability += probabilities[i];
            if (random <= cumulativeProbability) {
                return actions[i];
            }
        }

        return actions[actions.length - 1]; // Fallback
    };

    const handleGetStrategy = () => {
        const action = getRandomAction();
        setSelectedAction(action);
    };

    if (!strategy) {
        return (
            <div style={{ color: 'white', textAlign: 'center', margin: '20px' }}>
                <p>No strategy found for this situation.</p>
                <p>Try adjusting your inputs or check if the situation exists in the training data.</p>
            </div>
        );
    }

    return (
        <div style={{ color: 'white', margin: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
            <h3>Mixed Strategy</h3>

            <div style={{ marginBottom: '20px' }}>
                <h4>Legal Actions:</h4>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                    {strategy.legal_actions.map(action => (
                        <span key={action} style={{
                            backgroundColor: '#555',
                            padding: '5px 10px',
                            borderRadius: '3px',
                            fontSize: '12px'
                        }}>
                            {action}
                        </span>
                    ))}
                </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <h4>Strategy Percentages:</h4>
                <div style={{ display: 'grid', gap: '10px' }}>
                    {Object.entries(strategy.average_strategy).map(([action, probability]) => (
                        <div key={action} style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            padding: '10px',
                            backgroundColor: '#333',
                            borderRadius: '5px'
                        }}>
                            <span style={{ fontWeight: 'bold' }}>{action}</span>
                            <span>{(probability * 100).toFixed(1)}%</span>
                            <div style={{
                                width: '100px',
                                height: '10px',
                                backgroundColor: '#555',
                                borderRadius: '5px',
                                overflow: 'hidden'
                            }}>
                                <div style={{
                                    width: `${probability * 100}%`,
                                    height: '100%',
                                    backgroundColor: '#4CAF50'
                                }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <button
                onClick={handleGetStrategy}
                style={{
                    padding: '10px 20px',
                    fontSize: '16px',
                    backgroundColor: '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer',
                    width: '100%'
                }}
            >
                Get Random Strategy
            </button>

            {selectedAction && (
                <div style={{
                    marginTop: '20px',
                    padding: '15px',
                    backgroundColor: '#4CAF50',
                    borderRadius: '5px',
                    textAlign: 'center',
                    fontSize: '18px',
                    fontWeight: 'bold'
                }}>
                    Recommended Action: {selectedAction.toUpperCase()}
                </div>
            )}
        </div>
    );
}

export default StrategyDisplay;

// C:\Ron\AllIn\frontend\src\components\CardInput.jsx
import React from 'react';

function CardInput({ gameState, setGameState }) {
    const updateHoleCards = (cardIndex, value) => {
        const newHoleCards = [...gameState.holeCards];
        newHoleCards[cardIndex] = value.toUpperCase();
        setGameState(prev => ({ ...prev, holeCards: newHoleCards }));
    };

    const updateCommunityCards = (cardIndex, value) => {
        const newCommunityCards = [...gameState.communityCards];
        newCommunityCards[cardIndex] = value.toUpperCase();

        // Just update the cards, don't change the street at all
        setGameState(prev => ({
            ...prev,
            communityCards: newCommunityCards
            // No street change - user controls it manually with buttons
        }));
    };

    const updateStreet = (street) => {
        setGameState(prev => ({ ...prev, street }));
    };

    return (
        <div style={{ color: 'white', margin: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
            <h3>Cards</h3>

            {/* Hole Cards */}
            <div style={{ marginBottom: '20px' }}>
                <label>Your Hole Cards:</label>
                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                    {[0, 1].map(index => (
                        <input
                            key={index}
                            type="text"
                            placeholder={index === 0 ? "AS" : "KH"}
                            value={gameState.holeCards[index] || ''}
                            onChange={(e) => updateHoleCards(index, e.target.value)}
                            style={{
                                width: '50px',
                                padding: '8px',
                                textAlign: 'center',
                                fontSize: '14px',
                                textTransform: 'uppercase'
                            }}
                            maxLength={2}
                        />
                    ))}
                </div>
            </div>

            {/* Street Selection */}
            <div style={{ marginBottom: '20px' }}>
                <label>Street:</label>
                <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                    {['preflop', 'flop', 'turn', 'river'].map(street => (
                        <button
                            key={street}
                            onClick={() => updateStreet(street)}
                            style={{
                                padding: '8px 16px',
                                backgroundColor: gameState.street === street ? '#4CAF50' : '#666',
                                color: 'white',
                                border: 'none',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                textTransform: 'capitalize'
                            }}
                        >
                            {street}
                        </button>
                    ))}
                </div>
            </div>

            {/* Community Cards */}
            {gameState.street !== 'preflop' && (
                <div style={{ marginBottom: '20px' }}>
                    <label>Community Cards:</label>
                    <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                        {[0, 1, 2, 3, 4].map(index => {
                            const isVisible = (
                                (gameState.street === 'flop' && index < 3) ||
                                (gameState.street === 'turn' && index < 4) ||
                                (gameState.street === 'river' && index < 5)
                            );

                            return isVisible ? (
                                <input
                                    key={index}
                                    type="text"
                                    placeholder={["QD", "JC", "TC", "9H", "8S"][index]}
                                    value={gameState.communityCards[index] || ''}
                                    onChange={(e) => updateCommunityCards(index, e.target.value)}
                                    style={{
                                        width: '50px',
                                        padding: '8px',
                                        textAlign: 'center',
                                        fontSize: '14px',
                                        textTransform: 'uppercase'
                                    }}
                                    maxLength={2}
                                />
                            ) : null;
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

export default CardInput;
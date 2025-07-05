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
        setGameState(prev => ({ ...prev, communityCards: newCommunityCards }));
    };

    const updateStreet = (street) => {
        setGameState(prev => ({ ...prev, street }));
    };

    return (
        <div style={{ color: 'white', margin: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '10px' }}>
            <h3>Cards</h3>

            {/* Hole Cards */}
            <div>
                <label>Your Hole Cards:</label>
                <div style={{ display: 'flex', gap: '10px', margin: '10px 0' }}>
                    <input
                        type="text"
                        placeholder="AH"
                        maxLength="2"
                        value={gameState.holeCards[0] || ''}
                        onChange={(e) => updateHoleCards(0, e.target.value)}
                        style={{ width: '50px', padding: '5px' }}
                    />
                    <input
                        type="text"
                        placeholder="KS"
                        maxLength="2"
                        value={gameState.holeCards[1] || ''}
                        onChange={(e) => updateHoleCards(1, e.target.value)}
                        style={{ width: '50px', padding: '5px' }}
                    />
                </div>
            </div>

            {/* Street Selection */}
            <div style={{ margin: '20px 0' }}>
                <label>Street:</label>
                <div style={{ display: 'flex', gap: '10px', margin: '10px 0' }}>
                    {['preflop', 'flop', 'turn', 'river'].map(street => (
                        <button
                            key={street}
                            onClick={() => updateStreet(street)}
                            style={{
                                padding: '5px 10px',
                                backgroundColor: gameState.street === street ? '#4CAF50' : '#ddd',
                                color: gameState.street === street ? 'white' : 'black',
                                border: 'none',
                                borderRadius: '3px',
                                cursor: 'pointer'
                            }}
                        >
                            {street.charAt(0).toUpperCase() + street.slice(1)}
                        </button>
                    ))}
                </div>
            </div>

            {/* Community Cards */}
            {gameState.street !== 'preflop' && (
                <div>
                    <label>Community Cards:</label>
                    <div style={{ display: 'flex', gap: '10px', margin: '10px 0' }}>
                        {[0, 1, 2, 3, 4].map(index => {
                            const maxCards = gameState.street === 'flop' ? 3 :
                                gameState.street === 'turn' ? 4 : 5;

                            if (index >= maxCards) return null;

                            return (
                                <input
                                    key={index}
                                    type="text"
                                    placeholder={`Card ${index + 1}`}
                                    maxLength="2"
                                    value={gameState.communityCards[index] || ''}
                                    onChange={(e) => updateCommunityCards(index, e.target.value)}
                                    style={{ width: '50px', padding: '5px' }}
                                />
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

export default CardInput;
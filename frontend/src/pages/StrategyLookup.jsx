// frontend/src/pages/StrategyLookup.jsx
import React, { useState, useEffect } from 'react';
import strategyData from '../../../backend/bot/analysis/blueprint.json';

function StrategyLookup() {
    const [gameState, setGameState] = useState({
        holeCards: ['', ''],
        communityCards: ['', '', '', '', ''],
        street: 'preflop',
        potSize: 1.5,
        actions: []
    });

    const [betInput, setBetInput] = useState('');
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Clear actions when street changes
    useEffect(() => {
        setGameState(prev => ({ ...prev, actions: [] }));
    }, [gameState.street]);

    // Strict card validation
    const validateAndFormatCard = (input) => {
        if (!input) return '';

        const cleaned = input.replace(/[^a-zA-Z0-9]/g, '');
        if (cleaned.length === 0) return '';

        if (cleaned.length === 1) {
            const rank = cleaned[0].toUpperCase();
            const validRanks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
            return validRanks.includes(rank) ? rank : '';
        }

        if (cleaned.length >= 2) {
            const rank = cleaned[0].toUpperCase();
            const suit = cleaned[1].toLowerCase();
            const validRanks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'];
            const validSuits = ['c', 'd', 's', 'h'];

            if (validRanks.includes(rank) && validSuits.includes(suit)) {
                return rank + suit;
            } else if (validRanks.includes(rank)) {
                return rank;
            }
            return '';
        }

        return '';
    };

    // Handle card input changes
    const handleCardChange = (cardType, index, value) => {
        const formattedValue = validateAndFormatCard(value);

        setGameState(prev => {
            const newState = { ...prev };
            if (cardType === 'hole') {
                newState.holeCards[index] = formattedValue;
            } else {
                newState.communityCards[index] = formattedValue;
            }
            return newState;
        });
    };

    // Check for duplicate cards
    const getDuplicateCards = () => {
        const allCards = [...gameState.holeCards, ...gameState.communityCards]
            .filter(card => card && card.length === 2);
        const cardCounts = {};
        const duplicates = new Set();

        allCards.forEach(card => {
            cardCounts[card] = (cardCounts[card] || 0) + 1;
            if (cardCounts[card] > 1) {
                duplicates.add(card);
            }
        });

        return duplicates;
    };

    // Clear functions
    const clearHoleCards = () => {
        setGameState(prev => ({ ...prev, holeCards: ['', ''] }));
    };

    const clearCommunityCards = () => {
        setGameState(prev => ({ ...prev, communityCards: ['', '', '', '', ''] }));
    };

    // Calculate current pot in real-time - FIXED to use actual initial pot
    const calculateCurrentPot = (actions, initialPot) => {
        let currentPot = initialPot;

        for (let i = 0; i < actions.length; i++) {
            const action = actions[i];
            if (action.action === 'check' || action.action === 'fold') {
                continue;
            } else if (action.action === 'call') {
                const callAmount = getCallAmount(actions.slice(0, i), initialPot);
                currentPot += callAmount;
            } else if (action.action === 'bet' || action.action === 'raise') {
                currentPot += action.actualAmount;
            }
        }

        return currentPot;
    };

    // Get call amount
    const getCallAmount = (historyBeforeCall, initialPot) => {
        if (historyBeforeCall.length === 0) {
            return gameState.street === 'preflop' ? 0.5 : 0;
        }

        let lastBetAmount = 0;
        for (let i = historyBeforeCall.length - 1; i >= 0; i--) {
            const action = historyBeforeCall[i];
            if (action.action === 'bet' || action.action === 'raise') {
                lastBetAmount = action.totalContribution;
                break;
            }
        }

        const currentPlayer = historyBeforeCall.length % 2;
        let playerContribution = 0;

        if (gameState.street === 'preflop') {
            playerContribution = currentPlayer === 0 ? 0.5 : 1;
        }

        for (let i = 0; i < historyBeforeCall.length; i++) {
            const action = historyBeforeCall[i];
            const actionPlayer = i % 2;
            if (actionPlayer === currentPlayer && (action.action === 'bet' || action.action === 'raise')) {
                playerContribution = action.totalContribution;
            }
        }

        return Math.max(0, lastBetAmount - playerContribution);
    };

    // FIXED rounding logic - pass actual initial pot
    const roundBetToNearestSize = (inputAmount, street, actions, actualInitialPot) => {
        const inputInChips = inputAmount * 2; // Convert BB to chips

        if (street === 'preflop') {
            const betRaiseCount = actions.filter(a => a.action === 'bet' || a.action === 'raise').length;

            if (betRaiseCount === 0) {
                // First betting round: 3BB, 5BB, 7BB (6, 10, 14 chips)
                const sizes = [6, 10, 14];
                const sizeNames = ['small', 'medium', 'large'];
                return findClosestSize(inputInChips, sizes, sizeNames);
            } else if (betRaiseCount === 1) {
                // Second betting round: 6BB, 10BB, 14BB (12, 20, 28 chips)
                const sizes = [12, 20, 28];
                const sizeNames = ['small', 'medium', 'large'];
                return findClosestSize(inputInChips, sizes, sizeNames);
            } else {
                // Pot relative: 0.66x, 1.33x, 2.0x
                const currentPot = calculateCurrentPot(actions, actualInitialPot) * 2; // Convert to chips
                const sizes = [0.66 * currentPot, 1.33 * currentPot, 2.0 * currentPot];
                const sizeNames = ['small', 'medium', 'large'];
                return findClosestSize(inputInChips, sizes, sizeNames);
            }
        } else {
            // Postflop: 0.33x, 0.66x, 1.0x of pot - FIXED to use actual pot
            const currentPot = calculateCurrentPot(actions, actualInitialPot) * 2; // Convert to chips
            const sizes = [0.33 * currentPot, 0.66 * currentPot, 1.0 * currentPot];
            const sizeNames = ['small', 'medium', 'large'];
            return findClosestSize(inputInChips, sizes, sizeNames);
        }
    };

    const findClosestSize = (inputAmount, sizes, sizeNames) => {
        let closestIndex = 0;
        let closestDiff = Math.abs(inputAmount - sizes[0]);

        for (let i = 1; i < sizes.length; i++) {
            const diff = Math.abs(inputAmount - sizes[i]);
            if (diff < closestDiff) {
                closestDiff = diff;
                closestIndex = i;
            }
        }

        return {
            size: sizeNames[closestIndex],
            roundedAmount: sizes[closestIndex] / 2 // Convert back to BB
        };
    };

    // FIXED legal actions - proper check limiting and preflop logic
    const getLegalActions = () => {
        const actions = gameState.actions;

        // Check if fold in history
        if (actions.some(a => a.action === 'fold')) {
            return [];
        }

        // Check betting cap (1 bet + 3 raises = 4 total)
        const betRaiseCount = actions.filter(a => a.action === 'bet' || a.action === 'raise').length;
        if (betRaiseCount >= 4) {
            return [];
        }

        if (gameState.street === 'preflop') {
            return getPreflopLegalActions(actions);
        } else {
            return getPostflopLegalActions(actions);
        }
    };

    // FIXED preflop legal actions
    const getPreflopLegalActions = (actions) => {
        if (actions.length === 0) {
            // SB opening action
            return [
                { action: 'call', amount: 0.5 }, // SB calling BB
                { action: 'bet' }
            ];
        } else if (actions.length === 1) {
            // BB's turn after SB action
            const lastAction = actions[0];
            if (lastAction.action === 'call') {
                // SB called - BB can check or bet
                // NOTE: BB checking here ENDS the round, but we still allow it for strategy lookup
                return [
                    { action: 'check' },
                    { action: 'bet' }
                ];
            } else if (lastAction.action === 'bet') {
                // SB opened - BB can only raise (no call as it ends round)
                return [
                    { action: 'raise' }
                ];
            }
        }

        // Later actions - must handle check sequences properly
        const lastAction = actions[actions.length - 1];
        const secondLastAction = actions.length >= 2 ? actions[actions.length - 2] : null;

        // Check for round completion scenarios
        if (lastAction.action === 'check' && secondLastAction?.action === 'check') {
            return []; // Double check ends round
        }

        if (lastAction.action === 'call') {
            return []; // Call after bet/raise ends round
        }

        if (lastAction.action === 'check') {
            return [
                { action: 'check' },
                { action: 'bet' }
            ];
        } else if (lastAction.action === 'bet' || lastAction.action === 'raise') {
            const raiseCount = actions.filter(a => a.action === 'raise').length;

            if (raiseCount < 3) {
                return [{ action: 'raise' }];
            }
            return [];
        }

        return [];
    };

    // FIXED postflop legal actions - proper check limiting
    const getPostflopLegalActions = (actions) => {
        if (actions.length === 0) {
            // First action postflop
            return [
                { action: 'check' },
                { action: 'bet' }
            ];
        }

        const lastAction = actions[actions.length - 1];
        const secondLastAction = actions.length >= 2 ? actions[actions.length - 2] : null;

        // Check for round completion
        if (lastAction.action === 'check' && secondLastAction?.action === 'check') {
            return []; // Double check ends round
        }

        if (lastAction.action === 'call') {
            return []; // Call after bet/raise ends round
        }

        if (lastAction.action === 'check') {
            return [
                { action: 'check' },
                { action: 'bet' }
            ];
        } else if (lastAction.action === 'bet' || lastAction.action === 'raise') {
            const raiseCount = actions.filter(a => a.action === 'raise').length;

            if (raiseCount < 3) {
                return [{ action: 'raise' }];
            }
            return [];
        }

        return [];
    };

    // Add action with bet size - BETTER VALIDATION
    const addActionWithBetSize = (actionType) => {
        const inputAmount = parseFloat(betInput);
        if (isNaN(inputAmount) || inputAmount <= 0) {
            setError('Please enter a valid bet amount greater than 0');
            return;
        }

        // Calculate rounded size for info set generation - FIXED to pass actual pot
        const actualInitialPot = parseFloat(gameState.potSize) || 1.5;
        const rounded = roundBetToNearestSize(inputAmount, gameState.street, gameState.actions, actualInitialPot);

        // Calculate actual contribution logic
        const currentPlayer = gameState.actions.length % 2;
        let currentContribution = 0;

        if (gameState.street === 'preflop') {
            currentContribution = currentPlayer === 0 ? 0.5 : 1; // Blinds
        }

        // Find any previous bets/raises by this player
        for (let i = 0; i < gameState.actions.length; i++) {
            const action = gameState.actions[i];
            const actionPlayer = i % 2;
            if (actionPlayer === currentPlayer && (action.action === 'bet' || action.action === 'raise')) {
                currentContribution = action.totalContribution;
            }
        }

        const totalContribution = inputAmount; // User's input is total contribution
        const actualAmount = totalContribution - currentContribution; // Amount added to pot

        const newAction = {
            action: actionType,
            size: rounded.size,
            inputAmount: inputAmount,
            actualAmount: actualAmount,
            totalContribution: totalContribution
        };

        setGameState(prev => ({
            ...prev,
            actions: [...prev.actions, newAction]
        }));

        setBetInput('');
    };

    // Add simple action
    const addSimpleAction = (actionData) => {
        setGameState(prev => ({
            ...prev,
            actions: [...prev.actions, actionData]
        }));
    };

    // Remove last action
    const removeLastAction = () => {
        setGameState(prev => ({
            ...prev,
            actions: prev.actions.slice(0, -1)
        }));
    };

    // Clear all actions
    const clearAllActions = () => {
        setGameState(prev => ({
            ...prev,
            actions: []
        }));
    };

    // FIXED pot size change - allow decimals
    const handlePotSizeChange = (value) => {
        if (value === '') {
            setGameState(prev => ({ ...prev, potSize: '' }));
            return;
        }

        // Allow any positive number including decimals
        const numValue = parseFloat(value);
        if (!isNaN(numValue) && numValue > 0) {
            setGameState(prev => ({ ...prev, potSize: value })); // Keep as string to preserve decimals
        }
    };

    const handlePotSizeBlur = () => {
        // Ensure minimum value on blur
        const numValue = parseFloat(gameState.potSize);
        if (isNaN(numValue) || numValue < 1.5) {
            setGameState(prev => ({ ...prev, potSize: 1.5 }));
        }
    };

    // Validate inputs
    const validateInputs = () => {
        const validHoleCards = gameState.holeCards.filter(card => card && card.length === 2);
        if (validHoleCards.length < 2) {
            setError('Both hole cards are required');
            return false;
        }

        const duplicates = getDuplicateCards();
        if (duplicates.size > 0) {
            setError('Duplicate cards detected: ' + Array.from(duplicates).join(', '));
            return false;
        }

        const potValue = parseFloat(gameState.potSize);
        if (isNaN(potValue) || potValue < 1.5) {
            setError('Initial pot size must be at least 1.5BB');
            return false;
        }

        setError(null);
        return true;
    };

    // Main lookup function
    const lookupStrategy = async () => {
        if (!validateInputs()) return;

        setLoading(true);
        setResults(null);

        try {
            const response = await fetch('http://localhost:5000/api/evaluate-hand', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    holeCards: gameState.holeCards.filter(card => card && card.length === 2),
                    communityCards: gameState.communityCards.filter(card => card && card.length === 2),
                    actions: gameState.actions.map(a => ({
                        action: a.action,
                        amount: a.actualAmount ? a.actualAmount * 2 : 0,
                        size: a.size || null
                    })),
                    gameState: {
                        potSize: parseFloat(gameState.potSize) * 2,
                        initialPotSize: 3,
                        street: gameState.street
                    }
                })
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            const strategy = strategyData.normalized_strategies[data.infoSetKey];

            setResults({
                infoSetKey: data.infoSetKey,
                strategy: strategy,
                debugInfo: data.debugInfo,
                exists: !!strategy
            });

        } catch (err) {
            setError(`Lookup failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    const getBestAction = (strategy) => {
        if (!strategy?.average_strategy) return null;

        let maxProb = 0;
        let bestAction = null;

        Object.entries(strategy.average_strategy).forEach(([action, prob]) => {
            if (prob > maxProb) {
                maxProb = prob;
                bestAction = action;
            }
        });

        return { action: bestAction, probability: maxProb };
    };

    const legalActions = getLegalActions();
    const actualInitialPot = parseFloat(gameState.potSize) || 1.5;
    const currentPot = calculateCurrentPot(gameState.actions, actualInitialPot);
    const duplicateCards = getDuplicateCards();

    // Helper function to check if community card input should be enabled
    const isCommunityCardEnabled = (index) => {
        const streetLimits = { preflop: 0, flop: 3, turn: 4, river: 5 };
        return index < streetLimits[gameState.street];
    };

    return (
        <div style={{
            minHeight: '100vh',
            width: '100vw',
            background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
            color: 'white',
            fontFamily: '"Inter", sans-serif',
            padding: '0',
            margin: '0'
        }}>
            <div style={{
                width: '100%',
                padding: '1.5rem',
                boxSizing: 'border-box'
            }}>
                {/* Header */}
                <div style={{
                    textAlign: 'center',
                    marginBottom: '2rem'
                }}>
                    <h1 style={{
                        fontSize: '2.5rem',
                        fontWeight: '800',
                        background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        margin: '0 0 0.5rem 0'
                    }}>
                        Strategy Lookup
                    </h1>
                    <p style={{
                        fontSize: '1rem',
                        color: '#94a3b8',
                        margin: '0'
                    }}>
                        Current Pot: {currentPot.toFixed(2)}BB | Street: {gameState.street} | Actions: {gameState.actions.length}
                    </p>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr 1fr',
                    gap: '1.5rem',
                    marginBottom: '2rem',
                    minHeight: 'calc(100vh - 200px)'
                }}>
                    {/* Cards & Game State */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '1.5rem'
                    }}>
                        {/* Hole Cards */}
                        <div style={{
                            background: 'rgba(15, 23, 42, 0.8)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(59, 130, 246, 0.2)',
                            borderRadius: '1rem',
                            padding: '1.5rem'
                        }}>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: '1rem'
                            }}>
                                <h3 style={{
                                    color: '#3b82f6',
                                    fontSize: '1.125rem',
                                    fontWeight: '600',
                                    margin: '0'
                                }}>
                                    🃏 Hole Cards
                                </h3>
                                <button
                                    onClick={clearHoleCards}
                                    style={{
                                        padding: '0.25rem 0.5rem',
                                        background: 'rgba(239, 68, 68, 0.2)',
                                        border: '1px solid #ef4444',
                                        borderRadius: '0.25rem',
                                        color: '#ef4444',
                                        fontSize: '0.75rem',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Clear
                                </button>
                            </div>
                            <div style={{ display: 'flex', gap: '0.75rem' }}>
                                {gameState.holeCards.map((card, idx) => (
                                    <input
                                        key={`hole-${idx}`}
                                        type="text"
                                        value={card}
                                        onChange={(e) => handleCardChange('hole', idx, e.target.value)}
                                        placeholder={`Card ${idx + 1}`}
                                        style={{
                                            width: '70px',
                                            height: '50px',
                                            textAlign: 'center',
                                            fontSize: '1.125rem',
                                            fontWeight: '700',
                                            border: `2px solid ${duplicateCards.has(card) ? '#ef4444' :
                                                    card && card.length === 2 ? '#10b981' :
                                                        card ? '#ef4444' : '#475569'
                                                }`,
                                            borderRadius: '0.5rem',
                                            background: 'rgba(30, 41, 59, 0.8)',
                                            color: 'white',
                                            outline: 'none'
                                        }}
                                        maxLength={2}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* Community Cards */}
                        <div style={{
                            background: 'rgba(15, 23, 42, 0.8)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(59, 130, 246, 0.2)',
                            borderRadius: '1rem',
                            padding: '1.5rem'
                        }}>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: '1rem'
                            }}>
                                <h3 style={{
                                    color: '#3b82f6',
                                    fontSize: '1.125rem',
                                    fontWeight: '600',
                                    margin: '0'
                                }}>
                                    🎯 Community Cards
                                </h3>
                                <button
                                    onClick={clearCommunityCards}
                                    style={{
                                        padding: '0.25rem 0.5rem',
                                        background: 'rgba(239, 68, 68, 0.2)',
                                        border: '1px solid #ef4444',
                                        borderRadius: '0.25rem',
                                        color: '#ef4444',
                                        fontSize: '0.75rem',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Clear
                                </button>
                            </div>
                            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                                {gameState.communityCards.map((card, idx) => {
                                    const isEnabled = isCommunityCardEnabled(idx);

                                    return (
                                        <input
                                            key={`community-${idx}`}
                                            type="text"
                                            value={card}
                                            onChange={(e) => handleCardChange('community', idx, e.target.value)}
                                            placeholder={['F1', 'F2', 'F3', 'T', 'R'][idx]}
                                            disabled={!isEnabled}
                                            style={{
                                                width: '50px',
                                                height: '40px',
                                                textAlign: 'center',
                                                fontSize: '0.875rem',
                                                fontWeight: '600',
                                                border: `2px solid ${!isEnabled ? '#374151' :
                                                        duplicateCards.has(card) ? '#ef4444' :
                                                            card && card.length === 2 ? '#10b981' :
                                                                card ? '#ef4444' : '#475569'
                                                    }`,
                                                borderRadius: '0.5rem',
                                                background: !isEnabled ? 'rgba(15, 23, 42, 0.5)' : 'rgba(30, 41, 59, 0.8)',
                                                color: !isEnabled ? '#6b7280' : 'white',
                                                outline: 'none',
                                                opacity: !isEnabled ? 0.5 : 1,
                                                cursor: !isEnabled ? 'not-allowed' : 'text'
                                            }}
                                            maxLength={2}
                                        />
                                    );
                                })}
                            </div>

                            {/* Street and Pot Size */}
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                <div>
                                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: '500', color: '#e2e8f0', marginBottom: '0.25rem' }}>
                                        Street:
                                    </label>
                                    <select
                                        value={gameState.street}
                                        onChange={(e) => setGameState(prev => ({ ...prev, street: e.target.value }))}
                                        style={{
                                            width: '100%',
                                            padding: '0.5rem',
                                            border: '2px solid #475569',
                                            borderRadius: '0.5rem',
                                            background: 'rgba(30, 41, 59, 0.8)',
                                            color: 'white',
                                            fontSize: '0.875rem'
                                        }}
                                    >
                                        <option value="preflop">Preflop</option>
                                        <option value="flop">Flop</option>
                                        <option value="turn">Turn</option>
                                        <option value="river">River</option>
                                    </select>
                                </div>

                                <div>
                                    <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: '500', color: '#e2e8f0', marginBottom: '0.25rem' }}>
                                        Initial Pot (BB):
                                    </label>
                                    <input
                                        type="text"
                                        value={gameState.potSize}
                                        onChange={(e) => handlePotSizeChange(e.target.value)}
                                        onBlur={handlePotSizeBlur}
                                        placeholder="1.5"
                                        style={{
                                            width: '100%',
                                            padding: '0.5rem',
                                            border: `2px solid ${isNaN(parseFloat(gameState.potSize)) || parseFloat(gameState.potSize) < 1.5 ? '#ef4444' : '#475569'
                                                }`,
                                            borderRadius: '0.5rem',
                                            background: 'rgba(30, 41, 59, 0.8)',
                                            color: 'white',
                                            fontSize: '0.875rem'
                                        }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Action Buttons & Bet Input */}
                    <div style={{
                        background: 'rgba(15, 23, 42, 0.8)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(16, 185, 129, 0.2)',
                        borderRadius: '1rem',
                        padding: '1.5rem'
                    }}>
                        <h3 style={{
                            color: '#10b981',
                            fontSize: '1.125rem',
                            fontWeight: '600',
                            margin: '0 0 1rem 0'
                        }}>
                            🎮 Available Actions
                        </h3>

                        <div style={{ display: 'grid', gap: '0.75rem', marginBottom: '1.5rem' }}>
                            {legalActions.map((actionData, idx) => (
                                <button
                                    key={idx}
                                    onClick={() => addSimpleAction(actionData)}
                                    style={{
                                        padding: '0.75rem 1rem',
                                        background: actionData.action === 'call' || actionData.action === 'check' ? 'rgba(59, 130, 246, 0.2)' :
                                            'rgba(16, 185, 129, 0.2)',
                                        border: `2px solid ${actionData.action === 'call' || actionData.action === 'check' ? '#3b82f6' :
                                            '#10b981'}`,
                                        borderRadius: '0.5rem',
                                        color: 'white',
                                        fontSize: '0.875rem',
                                        fontWeight: '600',
                                        cursor: 'pointer',
                                        textAlign: 'left',
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center'
                                    }}
                                >
                                    <span>
                                        {actionData.action.toUpperCase()}
                                    </span>
                                    {actionData.amount !== undefined && (
                                        <span style={{ opacity: 0.8 }}>
                                            {actionData.amount.toFixed(2)}BB
                                        </span>
                                    )}
                                </button>
                            ))}
                        </div>

                        {/* Inline Bet/Raise Input */}
                        {legalActions.some(a => a.action === 'bet' || a.action === 'raise') && (
                            <div style={{
                                background: 'rgba(30, 41, 59, 0.5)',
                                border: '2px solid #3b82f6',
                                borderRadius: '0.75rem',
                                padding: '1rem',
                                marginBottom: '1.5rem'
                            }}>
                                <h4 style={{ color: '#3b82f6', margin: '0 0 1rem 0' }}>
                                    {legalActions.find(a => a.action === 'bet') ? 'Bet' : 'Raise'} Amount (BB):
                                </h4>
                                <div style={{ marginBottom: '0.75rem' }}>
                                    <small style={{ color: '#94a3b8', display: 'block' }}>
                                        Current pot: {currentPot.toFixed(2)}BB
                                    </small>
                                    <small style={{ color: '#94a3b8', display: 'block' }}>
                                        {gameState.street === 'preflop'
                                            ? `Sizes: ${gameState.actions.filter(a => a.action === 'bet' || a.action === 'raise').length === 0
                                                ? '3BB, 5BB, 7BB' : gameState.actions.filter(a => a.action === 'bet' || a.action === 'raise').length === 1
                                                    ? '6BB, 10BB, 14BB' : 'Pot relative'}`
                                            : `Sizes: ${(0.33 * currentPot).toFixed(1)}BB, ${(0.66 * currentPot).toFixed(1)}BB, ${(1.0 * currentPot).toFixed(1)}BB`
                                        }
                                    </small>
                                </div>
                                <div style={{ display: 'flex', gap: '0.5rem' }}>
                                    <input
                                        type="number"
                                        value={betInput}
                                        onChange={(e) => setBetInput(e.target.value)}
                                        placeholder="Enter BB amount..."
                                        step="0.1"
                                        style={{
                                            flex: 1,
                                            padding: '0.75rem',
                                            border: '1px solid #475569',
                                            borderRadius: '0.5rem',
                                            background: 'rgba(15, 23, 42, 0.8)',
                                            color: 'white',
                                            fontSize: '1rem'
                                        }}
                                    />
                                    <button
                                        onClick={() => addActionWithBetSize(legalActions.find(a => a.action === 'bet' || a.action === 'raise').action)}
                                        disabled={!betInput || parseFloat(betInput) <= 0 || isNaN(parseFloat(betInput))}
                                        style={{
                                            padding: '0.75rem 1rem',
                                            background: (!betInput || parseFloat(betInput) <= 0 || isNaN(parseFloat(betInput))) ? '#6b7280' : '#10b981',
                                            border: 'none',
                                            borderRadius: '0.5rem',
                                            color: 'white',
                                            fontWeight: '600',
                                            cursor: (!betInput || parseFloat(betInput) <= 0 || isNaN(parseFloat(betInput))) ? 'not-allowed' : 'pointer'
                                        }}
                                    >
                                        Add
                                    </button>
                                </div>
                                {betInput && parseFloat(betInput) > 0 && !isNaN(parseFloat(betInput)) && (
                                    <div style={{ marginTop: '0.5rem' }}>
                                        <small style={{ color: '#10b981' }}>
                                            Will round to: {roundBetToNearestSize(parseFloat(betInput), gameState.street, gameState.actions, actualInitialPot).size}
                                            ({roundBetToNearestSize(parseFloat(betInput), gameState.street, gameState.actions, actualInitialPot).roundedAmount.toFixed(2)}BB)
                                        </small>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Action Controls */}
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            <button
                                onClick={removeLastAction}
                                disabled={gameState.actions.length === 0}
                                style={{
                                    flex: 1,
                                    padding: '0.5rem',
                                    background: 'rgba(251, 191, 36, 0.2)',
                                    border: '2px solid #fbbf24',
                                    borderRadius: '0.5rem',
                                    color: 'white',
                                    fontSize: '0.75rem',
                                    fontWeight: '600',
                                    cursor: gameState.actions.length === 0 ? 'not-allowed' : 'pointer',
                                    opacity: gameState.actions.length === 0 ? 0.5 : 1
                                }}
                            >
                                Undo
                            </button>
                            <button
                                onClick={clearAllActions}
                                disabled={gameState.actions.length === 0}
                                style={{
                                    flex: 1,
                                    padding: '0.5rem',
                                    background: 'rgba(239, 68, 68, 0.2)',
                                    border: '2px solid #ef4444',
                                    borderRadius: '0.5rem',
                                    color: 'white',
                                    fontSize: '0.75rem',
                                    fontWeight: '600',
                                    cursor: gameState.actions.length === 0 ? 'not-allowed' : 'pointer',
                                    opacity: gameState.actions.length === 0 ? 0.5 : 1
                                }}
                            >
                                Clear All
                            </button>
                        </div>
                    </div>

                    {/* Action History & Lookup */}
                    <div style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '1.5rem'
                    }}>
                        {/* Action History */}
                        <div style={{
                            background: 'rgba(15, 23, 42, 0.8)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(139, 92, 246, 0.2)',
                            borderRadius: '1rem',
                            padding: '1.5rem'
                        }}>
                            <h3 style={{
                                color: '#8b5cf6',
                                fontSize: '1.125rem',
                                fontWeight: '600',
                                margin: '0 0 1rem 0'
                            }}>
                                📝 Action History
                            </h3>

                            <div style={{
                                minHeight: '200px',
                                maxHeight: '300px',
                                overflowY: 'auto',
                                background: 'rgba(30, 41, 59, 0.5)',
                                borderRadius: '0.5rem',
                                padding: '0.75rem'
                            }}>
                                {gameState.actions.length === 0 ? (
                                    <p style={{ color: '#94a3b8', textAlign: 'center', margin: '2rem 0' }}>
                                        No actions yet. Use the buttons to add actions.
                                    </p>
                                ) : (
                                    gameState.actions.map((action, idx) => (
                                        <div key={idx} style={{
                                            display: 'flex',
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            padding: '0.5rem',
                                            background: idx === gameState.actions.length - 1 ? 'rgba(59, 130, 246, 0.1)' : 'transparent',
                                            borderRadius: '0.25rem',
                                            marginBottom: '0.25rem'
                                        }}>
                                            <div style={{ color: '#e2e8f0' }}>
                                                <div>
                                                    {idx + 1}. {action.action.toUpperCase()}
                                                    {action.size ? ` (${action.size})` : ''}
                                                </div>
                                                {action.inputAmount && (
                                                    <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                                                        Input: {action.inputAmount}BB → Rounded: {action.size}
                                                    </div>
                                                )}
                                            </div>
                                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'end' }}>
                                                {action.actualAmount !== undefined && (
                                                    <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                                                        +{action.actualAmount.toFixed(2)}BB
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </div>

                        {/* Lookup Button */}
                        <button
                            onClick={lookupStrategy}
                            disabled={loading}
                            style={{
                                padding: '1rem',
                                fontSize: '1.125rem',
                                fontWeight: '600',
                                background: loading ? 'rgba(107, 114, 128, 0.8)' : 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                                color: 'white',
                                border: 'none',
                                borderRadius: '0.75rem',
                                cursor: loading ? 'not-allowed' : 'pointer',
                                opacity: loading ? 0.6 : 1,
                                boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
                            }}
                        >
                            {loading ? '🔍 Analyzing...' : '🚀 Lookup Strategy'}
                        </button>
                    </div>
                </div>

                {/* Error Display */}
                {error && (
                    <div style={{
                        backgroundColor: '#e53e3e',
                        color: 'white',
                        padding: '1rem',
                        borderRadius: '0.5rem',
                        marginBottom: '1.5rem',
                        textAlign: 'center'
                    }}>
                        <strong>Error:</strong> {error}
                    </div>
                )}

                {/* Results */}
                {results && (
                    <div style={{
                        background: 'rgba(15, 23, 42, 0.8)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(59, 130, 246, 0.2)',
                        borderRadius: '1rem',
                        padding: '2rem'
                    }}>
                        <h3 style={{
                            color: '#3b82f6',
                            fontSize: '1.25rem',
                            fontWeight: '700',
                            marginBottom: '1.5rem'
                        }}>
                            📊 Strategy Results
                        </h3>

                        {results.exists ? (
                            <div>
                                {/* Best Action */}
                                {(() => {
                                    const bestAction = getBestAction(results.strategy);
                                    return bestAction ? (
                                        <div style={{
                                            background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%)',
                                            border: '2px solid #10b981',
                                            borderRadius: '1rem',
                                            padding: '1.5rem',
                                            marginBottom: '2rem',
                                            textAlign: 'center'
                                        }}>
                                            <h4 style={{ color: '#10b981', marginBottom: '0.5rem' }}>🎯 Recommended Action</h4>
                                            <div style={{ fontSize: '1.5rem', fontWeight: '800', color: '#10b981' }}>
                                                {bestAction.action.replace('_', ' ').toUpperCase()}
                                            </div>
                                            <div style={{ color: '#6ee7b7', fontSize: '0.875rem' }}>
                                                {(bestAction.probability * 100).toFixed(1)}% probability
                                            </div>
                                        </div>
                                    ) : null;
                                })()}

                                {/* Strategy Breakdown */}
                                <div style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                                    gap: '1rem'
                                }}>
                                    {Object.entries(results.strategy.average_strategy)
                                        .sort(([, a], [, b]) => b - a)
                                        .map(([action, prob]) => (
                                            <div key={action} style={{
                                                background: 'rgba(30, 41, 59, 0.5)',
                                                borderRadius: '0.5rem',
                                                padding: '1rem',
                                                border: prob > 0.5 ? '1px solid #10b981' : prob > 0.2 ? '1px solid #3b82f6' : '1px solid #475569'
                                            }}>
                                                <div style={{ color: '#e2e8f0', fontWeight: '600', marginBottom: '0.5rem' }}>
                                                    {action.replace('_', ' ').toUpperCase()}
                                                </div>
                                                <div style={{
                                                    width: '100%',
                                                    height: '6px',
                                                    background: 'rgba(71, 85, 105, 0.5)',
                                                    borderRadius: '3px',
                                                    overflow: 'hidden',
                                                    marginBottom: '0.5rem'
                                                }}>
                                                    <div style={{
                                                        width: `${prob * 100}%`,
                                                        height: '100%',
                                                        background: prob > 0.5 ? '#10b981' : prob > 0.2 ? '#3b82f6' : '#8b5cf6'
                                                    }} />
                                                </div>
                                                <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                                                    {(prob * 100).toFixed(1)}%
                                                </div>
                                            </div>
                                        ))}
                                </div>
                            </div>
                        ) : (
                            <div style={{
                                background: 'rgba(239, 68, 68, 0.1)',
                                border: '2px solid #ef4444',
                                borderRadius: '1rem',
                                padding: '2rem',
                                textAlign: 'center'
                            }}>
                                <h4 style={{ color: '#ef4444', marginBottom: '1rem' }}>🚫 Strategy Not Found</h4>
                                <p style={{ color: '#fca5a5' }}>
                                    This situation was not encountered during training.
                                    Info set key: <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 4px', borderRadius: '2px' }}>{results.infoSetKey}</code>
                                </p>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* CSS to hide number input spinners */}
            <style>{`
        input[type="number"]::-webkit-outer-spin-button,
        input[type="number"]::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }
        input[type="number"] {
          -moz-appearance: textfield;
        }
      `}</style>
        </div>
    );
}

export default StrategyLookup;

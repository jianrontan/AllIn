// frontend/src/components/PlayingCard.jsx
// Renders one card in display format ('Ah', 'Td', '2c'), or a face-down back.
import React from 'react';

const SUITS = {
    h: { sym: '♥', red: true },   // hearts
    d: { sym: '♦', red: true },   // diamonds
    c: { sym: '♣', red: false },  // clubs
    s: { sym: '♠', red: false },  // spades
};

function PlayingCard({ card, hidden, small }) {
    const box = small ? 'w-11 h-16' : 'w-14 h-20';

    if (hidden || !card) {
        return (
            <div className={`${box} rounded-lg flex items-center justify-center
                bg-gradient-to-br from-slate-700 to-slate-900
                ring-1 ring-inset ring-amber-500/30`}>
                <div className="w-2.5 h-2.5 rotate-45 bg-amber-500/40" />
            </div>
        );
    }

    const rank = card[0] === 'T' ? '10' : card[0].toUpperCase();
    const suit = SUITS[card[1]] || { sym: '?', red: false };

    return (
        <div className={`${box} rounded-lg bg-white shadow-md flex flex-col
            items-center justify-center leading-none
            ${suit.red ? 'text-rose-600' : 'text-slate-900'}`}>
            <span className={small ? 'text-base font-bold' : 'text-xl font-bold'}>
                {rank}
            </span>
            <span className={small ? 'text-lg' : 'text-2xl'}>{suit.sym}</span>
        </div>
    );
}

export default PlayingCard;

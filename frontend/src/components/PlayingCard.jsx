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
    // The card sizes off the `--card-w` CSS variable set on the table container,
    // which is clamped to the viewport — so the whole board grows/shrinks with the
    // window. Font size is derived from the width, so the rank/suit scale with the
    // card. `small` pins a fixed compact size for use outside the table; elsewhere
    // the variable falls back to 3.5rem (the original size) so other pages are
    // unchanged.
    const sizeStyle = small
        ? { width: '2.5rem', height: '3.6rem', fontSize: '0.62rem' }
        : {
            width: 'var(--card-w, 3.5rem)',
            aspectRatio: '7 / 10',
            fontSize: 'calc(var(--card-w, 3.5rem) * 0.34)',
        };

    if (hidden || !card) {
        return (
            <div style={sizeStyle}
                className="rounded-lg flex items-center justify-center
                    bg-gradient-to-br from-slate-700 to-slate-900
                    ring-1 ring-inset ring-amber-500/30">
                <div className="w-[26%] aspect-square rotate-45 bg-amber-500/40" />
            </div>
        );
    }

    const rank = card[0] === 'T' ? '10' : card[0].toUpperCase();
    const suit = SUITS[card[1]] || { sym: '?', red: false };

    return (
        <div style={sizeStyle}
            className={`rounded-lg bg-white shadow-md flex flex-col
                items-center justify-center leading-none
                ${suit.red ? 'text-rose-600' : 'text-slate-900'}`}>
            <span className="font-bold" style={{ fontSize: '1em' }}>{rank}</span>
            <span style={{ fontSize: '1.28em' }}>{suit.sym}</span>
        </div>
    );
}

export default PlayingCard;

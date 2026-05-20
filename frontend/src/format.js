// frontend/src/format.js
// The backend works in chips (SB = 1 chip, BB = 2 chips). The UI shows
// everything in big blinds for consistency. One place to convert.

export const CHIPS_PER_BB = 2;

// chips -> big blinds, as a tidy string ("12" or "12.5", no trailing ".0")
export function fmtBB(chips) {
    const v = (chips || 0) / CHIPS_PER_BB;
    return Number.isInteger(v) ? String(v) : v.toFixed(1);
}

// signed variant, for profit/loss ("+12.5" / "-3")
export function fmtBBSigned(chips) {
    const s = fmtBB(chips);
    return chips > 0 ? `+${s}` : s;
}

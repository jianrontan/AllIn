// frontend/src/announcements.jsx
// Site announcements, NEWEST FIRST. To post one, prepend an entry with a fresh
// `id` and a `date` string. Everyone whose stored "last seen" id differs from the
// new top entry auto-sees the modal on their next visit; they can reopen it any
// time (the speaker button) to scroll back through older announcements.
import React from 'react';

// Render a bullet list from plain strings (kept as a helper, not a component, so
// this data-only module doesn't trip the fast-refresh "components only" rule).
const bullets = (points) => (
    <ul className="space-y-2.5">
        {points.map((t, i) => (
            <li key={i} className="flex gap-2">
                <span className="text-amber-500 shrink-0">•</span>
                <span>{t}</span>
            </li>
        ))}
    </ul>
);

export const ANNOUNCEMENTS = [
    {
        id: 'leaderboard-2026-06-23',
        date: 'June 23, 2026',
        title: 'Bot v2 is live + a new leaderboard',
        body: (
            <>
                <p className="mb-3">A few things just shipped:</p>
                {bullets([
                    'Bot v2 is here: a 52.5M iteration retrain on a finer abstraction, so it '
                        + 'comes back sharper.',
                    'The Bot vs humans record now breaks down by bot version, so you can '
                        + 'compare v1 and v2.',
                    'Unfortunately turn solving requires either $1000s of dollars of offline '
                        + 'compute or >60s of loading time, so no turn solving for now :(',
                    'I also added a special hidden feature, so see if you can figure it out! '
                ])}
                <p className="mt-3">Good luck,</p>
                <p className="font-medium text-amber-300/90">Ron</p>
            </>
        ),
    },
    {
        id: 'botv2-2026-06-16',
        date: 'June 16, 2026',
        title: 'Coming soon: Bot v2',
        body: (
            <>
                <p className="mb-3">Firstly, thanks for playing! I have a few updates:</p>
                {bullets([
                    'Bot v2 in the next few days: a 90M iteration retrain on a finer '
                        + 'abstraction, so it comes back a little stronger.',
                    'A turn solver is in the works to deepen its play.',
                    'More stats are coming to the front page: compare Bot v1 and v2, plus some of your own.',
                    'Plus a few fixes along the way.',
                ])}
                <p className="mt-3">Stay tuned,</p>
                <p className="font-medium text-amber-300/90">Ron</p>
            </>
        ),
    },
];

// frontend/src/pages/StrategyLookup.jsx
// Strategy Explorer - two separate, self-contained tools:
//   Hand Explorer : enter real cards + a betting line.
//   Key Explorer  : build an info-set key from the abstraction buckets.
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import HandExplorer from '../components/HandExplorer';
import KeyExplorer from '../components/KeyExplorer';
import { AnnouncementsButton } from '../components/Announcements';

const TABS = [
    { id: 'hand', label: 'Hand Explorer' },
    { id: 'key', label: 'Key Explorer' },
];

function StrategyLookup() {
    const [tab, setTab] = useState('hand');

    return (
        <div className="min-h-screen
                        bg-[radial-gradient(ellipse_at_top,#0c2a1f_0%,#0a0a0a_60%)]">
            {/* Full-bleed top bar: Home top-left, announcements top-right (matching
                the other pages); the explorer content below stays centred. */}
            <div className="w-full px-6 pt-6 flex items-center justify-between">
                <Link to="/" className="text-sm text-amber-400 hover:text-amber-300">
                    ← Home
                </Link>
                <AnnouncementsButton />
            </div>

            <div className="flex justify-center">
                <div className="w-full max-w-3xl px-6 pb-10 pt-6">
                    <h1 className="text-3xl font-bold tracking-tight">
                        Strategy Explorer
                    </h1>
                    <p className="text-neutral-400 mt-1">
                        Inspect what the trained blueprint does in a given situation.
                    </p>

                    <div className="flex gap-1 mt-6 mb-7 border-b border-neutral-800">
                        {TABS.map((t) => (
                            <button key={t.id} onClick={() => setTab(t.id)}
                                className={'px-6 py-2.5 text-sm rounded-t-lg transition-colors ' +
                                    (tab === t.id
                                        ? 'bg-amber-500 text-neutral-950 font-semibold'
                                        : 'bg-neutral-900 text-neutral-400 hover:text-neutral-200')}>
                                {t.label}
                            </button>
                        ))}
                    </div>

                    {/* Each tool keeps its own state; switching tabs does not share it. */}
                    {tab === 'hand' && <HandExplorer />}
                    {tab === 'key' && <KeyExplorer />}
                </div>
            </div>
        </div>
    );
}

export default StrategyLookup;

// frontend/src/components/VersionFilter.jsx
// Shared bot-version filter dropdown (All versions / v1 / v2 / ...), styled to match the
// app's dark + amber theme. Renders nothing until at least one version is available.
import React from 'react';

function VersionFilter({ value, onChange, versions, className = '' }) {
    if (!versions || versions.length === 0) return null;
    return (
        <div className={`relative inline-block ${className}`}>
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="appearance-none rounded-lg border border-neutral-700 bg-neutral-800/80
                           text-neutral-200 text-xs pl-2.5 pr-6 py-1 cursor-pointer
                           hover:border-amber-500/50 focus:border-amber-500/70 focus:outline-none
                           focus:ring-1 focus:ring-amber-500/30 transition-colors">
                <option value="all">All versions</option>
                {versions.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
            {/* Custom chevron (the native arrow is removed via appearance-none). */}
            <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2
                             text-amber-400/70 text-[10px]">▾</span>
        </div>
    );
}

export default VersionFilter;

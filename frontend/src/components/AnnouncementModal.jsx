// frontend/src/components/AnnouncementModal.jsx
// "Announcements" modal: shows every announcement newest-first (scroll back for
// older ones), same visual style as the IntroModal help screen. This is the
// presentation only; the open/auto-open/"dismissed" state lives in
// Announcements.jsx (AnnouncementsProvider). `onClose(dontShow)` reports the
// "Don't show this again" checkbox. Content lives in src/announcements.jsx.
import React, { useEffect, useState } from 'react';

// Megaphone icon (lucide-style), reused by the top-bar "Announcements" button.
export function Megaphone({ className = 'w-3.5 h-3.5' }) {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
            <path d="m3 11 18-5v12L3 14v-3z" />
            <path d="M11.6 16.8a3 3 0 1 1-5.8-1.6" />
        </svg>
    );
}

function AnnouncementModal({ open, items = [], onClose }) {
    const [dontShow, setDontShow] = useState(false);
    // Reset the checkbox every time the modal opens. This component never unmounts (the provider
    // always renders it with open=false), so without this the box would stay checked from a prior
    // "Don't show again" and appear pre-checked on the next open.
    useEffect(() => { if (open) setDontShow(false); }, [open]);
    if (!open) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
            onClick={() => onClose(dontShow)}>
            <div className="w-full max-w-md max-h-[80vh] flex flex-col rounded-2xl
                            border border-amber-600/40 bg-neutral-900 shadow-2xl"
                onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center gap-2 px-6 pt-6 pb-3 shrink-0">
                    <Megaphone className="w-4 h-4 text-amber-300" />
                    <h2 className="text-xl font-bold text-amber-300">Announcements</h2>
                </div>
                {items.length === 0 ? (
                    <p className="px-6 pb-6 text-sm text-neutral-500">Nothing here yet.</p>
                ) : (
                    // Scrollable history: newest at the top, scroll down for older.
                    // flex-1 + min-h-0 lets this shrink within the max-h-[80vh] modal so
                    // it actually scrolls (a flex child won't shrink below its content
                    // height without min-h-0) instead of overflowing the viewport.
                    <div className="px-6 overflow-y-auto divide-y divide-neutral-800 min-h-0">
                        {items.map((a) => (
                            <article key={a.id} className="py-5">
                                <div className="text-[11px] uppercase tracking-wider text-neutral-500 mb-1">
                                    {a.date}
                                </div>
                                <h3 className="text-base font-semibold text-amber-200 mb-2">{a.title}</h3>
                                <div className="text-sm text-neutral-300">{a.body}</div>
                            </article>
                        ))}
                    </div>
                )}
                <div className="p-6 pt-4 shrink-0">
                    <label className="flex items-center gap-2 text-xs text-neutral-400
                                      cursor-pointer mb-3">
                        <input type="checkbox" checked={dontShow}
                            onChange={(e) => setDontShow(e.target.checked)}
                            className="h-4 w-4 accent-amber-500" />
                        Don&rsquo;t show this again
                    </label>
                    <button onClick={() => onClose(dontShow)}
                        className="w-full px-5 py-3 rounded-xl font-semibold bg-amber-500
                                   text-neutral-950 hover:bg-amber-400 transition-colors">
                        Got it
                    </button>
                </div>
            </div>
        </div>
    );
}

export default AnnouncementModal;

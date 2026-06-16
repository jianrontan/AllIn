// frontend/src/components/Announcements.jsx
// Announcements: a context provider that owns the modal + "seen" state (mounted
// once in the app root, App.jsx), plus a small header BUTTON dropped next to the
// account control on each page. Splitting them this way lets the trigger live in
// every page's top-right header while the modal/auto-open stays single-sourced.
import React, {
    createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react';
import { useLocation } from 'react-router-dom';
import AnnouncementModal, { Megaphone } from './AnnouncementModal';
import { ANNOUNCEMENTS } from '../announcements';

// Two independent flags, both keyed to the latest announcement id so a NEW
// announcement resets both:
//   dismissed -> set ONLY by "Don't show this again"; stops the Home auto-pop.
//   viewed    -> set whenever the modal is opened (auto or button); clears the dot.
const ANNOUNCE_DISMISS_KEY = 'allin_announce_dismissed';
const ANNOUNCE_VIEWED_KEY = 'allin_announce_viewed';
const LATEST_ID = ANNOUNCEMENTS[0]?.id || null;

const Ctx = createContext({ open: () => {}, hasUnseen: false });

export function AnnouncementsProvider({ children }) {
    const { pathname } = useLocation();
    const [open, setOpen] = useState(false);
    const [viewed, setViewed] = useState(() => localStorage.getItem(ANNOUNCE_VIEWED_KEY));
    const hasUnseen = !!LATEST_ID && viewed !== LATEST_ID;

    // Opening the modal (auto-pop OR the header button) marks the latest as viewed,
    // which clears the unseen dot. Stable identity so the effect/memo below don't
    // re-run needlessly.
    const openModal = useCallback(() => {
        setOpen(true);
        if (LATEST_ID && localStorage.getItem(ANNOUNCE_VIEWED_KEY) !== LATEST_ID) {
            localStorage.setItem(ANNOUNCE_VIEWED_KEY, LATEST_ID);
            setViewed(LATEST_ID);
        }
    }, []);

    // Auto-pop ONLY on the Home page (each visit), unless dismissed via "Don't show
    // this again". Re-runs on pathname change because this provider lives in the
    // persistent root layout (it doesn't re-mount on navigation like a page does).
    // The button still opens it manually on every page; only the AUTO-open is Home-only.
    useEffect(() => {
        if (LATEST_ID && pathname === '/'
            && localStorage.getItem(ANNOUNCE_DISMISS_KEY) !== LATEST_ID) {
            openModal();
        }
    }, [pathname, openModal]);

    // "Don't show this again" suppresses the Home auto-pop; a plain close does not.
    const close = (dontShow) => {
        setOpen(false);
        if (dontShow && LATEST_ID) {
            localStorage.setItem(ANNOUNCE_DISMISS_KEY, LATEST_ID);
        }
    };

    // Stable context value so consumers (the button) don't re-render on every
    // provider render (e.g. each route change).
    const value = useMemo(() => ({ open: openModal, hasUnseen }), [openModal, hasUnseen]);

    return (
        <Ctx.Provider value={value}>
            {children}
            <AnnouncementModal open={open} items={ANNOUNCEMENTS} onClose={close} />
        </Ctx.Provider>
    );
}

// Header trigger, styled like the "?" help button so it sits next to it / the
// sign-in control. Shows an amber dot when there's an unseen announcement.
export function AnnouncementsButton({ className = '' }) {
    const { open, hasUnseen } = useContext(Ctx);
    if (ANNOUNCEMENTS.length === 0) return null;
    return (
        <button onClick={open} title="Announcements" aria-label="Announcements"
            className={'relative w-6 h-6 rounded-full border border-neutral-700 '
                + 'text-neutral-400 hover:text-neutral-200 flex items-center '
                + 'justify-center shrink-0 ' + className}>
            <Megaphone />
            {hasUnseen && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full
                                 bg-amber-400 ring-2 ring-neutral-900" />
            )}
        </button>
    );
}

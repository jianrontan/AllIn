// frontend/src/pages/Home.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Logo from "./AllIn_Black_Centered.png";
import EvCounter from "../components/EvCounter";
import Leaderboard from "../components/Leaderboard";
import GoogleSignInButton from "../components/GoogleSignInButton";
import { AnnouncementsButton } from "../components/Announcements";
import { getPlayerId, getAccount } from "../api";

function Home() {
	// Lazy initializer: read the cached account synchronously on FIRST render so
	// a signed-in user never sees the "Sign in with Google" button flash for a
	// frame before the effect fires (localStorage reads are cheap + sync).
	const [account, setAccount] = useState(() => getAccount());

	useEffect(() => {
		getPlayerId();                       // ensure the anonymous id exists
		setAccount(getAccount());
	}, []);

	return (
		<div className="min-h-screen flex flex-col items-center px-6 py-6
		                bg-[radial-gradient(ellipse_at_50%_26%,#0c2a1f_0%,#0a0a0a_72%)]">
			{/* Header: sign-in / account in the true top-right corner (full-bleed,
			    like the Play-with-AI page). Players are anonymous until they sign in;
			    signing in is what puts them on the leaderboard. */}
			<div className="w-full flex justify-end items-center gap-3 mb-2">
				<AnnouncementsButton />
				<GoogleSignInButton registered={account?.isRegistered} handle={account?.handle} />
			</div>

			{/* Logo is height-capped (max-h) as well as width-capped so the hero +
			    CTAs stay above the fold on a short/normal viewport. */}
			<img src={Logo} alt="AllIn"
				className="w-[22rem] max-w-[80vw] max-h-[26vh] object-contain drop-shadow-lg" />
			{/* Always one line: `whitespace-nowrap` + a fluid font that shrinks on
			    narrow screens (and tighter letter-spacing on mobile) so the tagline
			    scales down instead of wrapping. */}
			<p className="mt-2 mb-6 uppercase text-amber-500/70 whitespace-nowrap
			              tracking-[0.14em] sm:tracking-[0.25em]"
				style={{ fontSize: 'clamp(0.6rem, 3vw, 0.875rem)' }}>
				Heads-up Texas Hold&rsquo;em &middot; CFR+ AI
			</p>

			<div className="w-full max-w-md mb-6">
				<EvCounter />
			</div>

			<div className="flex flex-col sm:flex-row gap-4 mb-6">
				<Link to="/ai-game"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide text-center
					           bg-amber-500 text-neutral-950 shadow-lg shadow-amber-900/40
					           hover:bg-amber-400 transition-colors">
					Play with AI
				</Link>
				<Link to="/strategy-lookup"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide text-center
					           border border-amber-500/40 text-amber-200
					           hover:bg-amber-500/10 transition-colors">
					Look up Strategy
				</Link>
			</div>

			{/* Methodology framing, between the CTAs and the leaderboard. */}
			<p className="mb-8 text-xs text-neutral-600 text-center">
				Trained with Monte&nbsp;Carlo Counterfactual Regret Minimization Algorithm
			</p>

			<div className="w-full max-w-2xl">
				<Leaderboard title="Leaderboard" accountsOnly minHands={50}
					note="Signed-in players with 50+ hands, ranked by BB/hand." />
			</div>
		</div>
	);
}

export default Home;

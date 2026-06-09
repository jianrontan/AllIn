// frontend/src/pages/Home.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Logo from "./AllIn_Black_Centered.png";
import EvCounter from "../components/EvCounter";
import Leaderboard from "../components/Leaderboard";
import GoogleSignInButton from "../components/GoogleSignInButton";
import { getPlayerId, getAccount } from "../api";

function Home() {
	const [account, setAccount] = useState(null);

	useEffect(() => {
		getPlayerId();                       // ensure the anonymous id exists
		setAccount(getAccount());
	}, []);

	return (
		<div className="min-h-screen flex flex-col items-center px-6 py-10
		                bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_72%)]">
			{/* Header: sign-in / account in the true top-right corner (full-bleed,
			    like the Play-with-AI page). Players are anonymous until they sign in;
			    signing in is what puts them on the leaderboard. */}
			<div className="w-full flex justify-end mb-4">
				<GoogleSignInButton registered={account?.isRegistered} handle={account?.handle} />
			</div>

			<img src={Logo} alt="AllIn" className="w-[27rem] max-w-[88vw] drop-shadow-lg" />
			<p className="mt-3 mb-8 text-sm tracking-[0.25em] uppercase text-amber-500/70">
				Heads-up Texas Hold&rsquo;em &middot; CFR+ AI
			</p>

			<div className="w-full max-w-md mb-8">
				<EvCounter />
			</div>

			<div className="flex flex-col sm:flex-row gap-4 mb-14">
				<Link to="/ai-game"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide
					           bg-amber-500 text-neutral-950 shadow-lg shadow-amber-900/40
					           hover:bg-amber-400 transition-colors">
					Play with AI
				</Link>
				<Link to="/strategy-lookup"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide
					           border border-amber-500/40 text-amber-200
					           hover:bg-amber-500/10 transition-colors">
					Look up Strategy
				</Link>
			</div>

			<div className="w-full max-w-2xl">
				<Leaderboard title="Leaderboard" accountsOnly minHands={50}
					note="Signed-in players with 50+ hands, ranked by BB/100." />
			</div>

			<p className="mt-14 text-xs text-neutral-600">
				Trained with Monte&nbsp;Carlo Counterfactual Regret Minimization
			</p>
		</div>
	);
}

export default Home;

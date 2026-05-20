// C:\Ron\AllIn\frontend\src\pages\Home.jsx
import React from "react";
import { Link } from "react-router-dom";
import Logo from "./AllIn_Black_Centered.png";

function Home() {
	return (
		<div className="min-h-screen flex flex-col items-center justify-center px-6
		                bg-[radial-gradient(ellipse_at_center,#0c2a1f_0%,#0a0a0a_72%)]">
			<img src={Logo} alt="AllIn" className="w-[27rem] max-w-[88vw] drop-shadow-lg" />

			<p className="mt-3 mb-12 text-sm tracking-[0.25em] uppercase text-amber-500/70">
				Heads-up Texas Hold&rsquo;em &middot; CFR+ AI
			</p>

			<div className="flex flex-col sm:flex-row gap-4">
				<Link
					to="/ai-game"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide
					           bg-amber-500 text-neutral-950 shadow-lg shadow-amber-900/40
					           hover:bg-amber-400 transition-colors"
				>
					Play with AI
				</Link>
				<Link
					to="/strategy-lookup"
					className="px-9 py-4 rounded-xl text-lg font-semibold tracking-wide
					           border border-amber-500/40 text-amber-200
					           hover:bg-amber-500/10 transition-colors"
				>
					Look up Strategy
				</Link>
			</div>

			<p className="mt-16 text-xs text-neutral-600">
				Trained with Monte&nbsp;Carlo Counterfactual Regret Minimization
			</p>
		</div>
	);
}

export default Home;

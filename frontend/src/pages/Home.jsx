// C:\Ron\AllIn\frontend\src\pages\Home.jsx
import React from "react";
import "../styles.css";
import Logo from "./ALLIn.png";

function Home() {
	return (
		<div className="website-container">
			<div className="logo-container">
				<img src={Logo} alt="ALL In Logo" className="logo" />
			</div>
			<div className="poker-table">
				<a href="/ai-game" className="play-button">
					Play with AI
				</a>
				<a href="/strategy-lookup" className="play-button">
					Look up Strategy
				</a>
			</div>
		</div>
	);
}

export default Home;
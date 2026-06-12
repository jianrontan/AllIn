// frontend/src/pages/NotFound.jsx
//
// Rendered in two cases:
//   1. The catch-all `path: '*'` route in App.jsx -- an unknown URL.
//   2. The router's `errorElement` -- any unhandled error in a route's render
//      or loader. We use `useRouteError` to surface the underlying status/code
//      when that's the case, otherwise we treat it as a plain 404.
import React from "react";
import { Link, useRouteError, isRouteErrorResponse } from "react-router-dom";

function NotFound() {
	const error = useRouteError();

	// Distinguish a router-thrown 404 from a generic render error so the copy
	// stays accurate. A bare visit to /no-such-page hits the catch-all route
	// (no error object); a 5xx loader failure goes through errorElement.
	let status = "404";
	let label = "Page not found";
	let detail = "The page you requested does not exist or has been moved.";

	if (error) {
		if (isRouteErrorResponse(error)) {
			status = String(error.status);
			label = error.statusText || label;
		} else {
			status = "Error";
			label = "Something went wrong";
			detail = "An unexpected error occurred while rendering this page.";
		}
	}

	return (
		<div className="min-h-screen flex items-center justify-center px-6
		                bg-[#0a0a0a] text-neutral-200">
			<div className="w-full max-w-xl">
				{/* Eyebrow: small, neutral, all-caps -- typographic anchor */}
				<div className="text-xs tracking-[0.2em] text-neutral-500 uppercase mb-6">
					{status}
				</div>

				<h1 className="text-2xl sm:text-3xl font-semibold text-neutral-100 mb-3">
					{label}
				</h1>

				<p className="text-neutral-400 text-base leading-relaxed mb-10
				              max-w-md">
					{detail}
				</p>

				{/* Hairline separator */}
				<div className="h-px bg-neutral-800 mb-8" />

				{/* CTAs: text-only, no decoration, separated by a thin divider.
				    Underlines on hover keep it readable without shouting. */}
				<div className="flex items-center gap-6 text-sm">
					<Link
						to="/"
						className="text-neutral-200 hover:text-white
						           underline-offset-4 hover:underline"
					>
						Return to home
					</Link>
					<span className="text-neutral-700">·</span>
					<a
						href="https://github.com/jianrontan/AllIn"
						target="_blank"
						rel="noopener noreferrer"
						className="text-neutral-400 hover:text-neutral-200
						           underline-offset-4 hover:underline"
					>
						View on GitHub
					</a>
				</div>
			</div>
		</div>
	);
}

export default NotFound;

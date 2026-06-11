// C:\Ron\AllIn\frontend\src\App.jsx
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Home from './pages/Home';
import AiGame from './pages/AiGame';
import StrategyLookup from './pages/StrategyLookup';
import AuthCallback from './pages/AuthCallback';

// Data router (createBrowserRouter) rather than the declarative <BrowserRouter>,
// so AiGame can use `useBlocker` to intercept in-app navigation away from a live
// hand (the browser Back button AND the Home link) and route it through the
// leave-this-hand confirm. The route table is otherwise unchanged.
const router = createBrowserRouter([
  { path: '/', element: <Home /> },
  { path: '/ai-game', element: <AiGame /> },
  { path: '/strategy-lookup', element: <StrategyLookup /> },
  { path: '/auth/callback', element: <AuthCallback /> },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;

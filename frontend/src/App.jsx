// C:\Ron\AllIn\frontend\src\App.jsx
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Home from './pages/Home';
import AiGame from './pages/AiGame';
import StrategyLookup from './pages/StrategyLookup';
import AuthCallback from './pages/AuthCallback';
import NotFound from './pages/NotFound';

// Data router (createBrowserRouter) rather than the declarative <BrowserRouter>,
// so AiGame can use `useBlocker` to intercept in-app navigation away from a live
// hand (the browser Back button AND the Home link) and route it through the
// leave-this-hand confirm. The route table is otherwise unchanged.
// `errorElement` on the parent catches anything thrown during render of a
// child route (loader failure, render exception, react-router-thrown 404).
// Without this, the data router shows its default "Unexpected Application
// Error" page that suggests adding an errorElement -- exactly this fix.
// The catch-all `path: '*'` covers the cleaner case where no route matches.
const router = createBrowserRouter([
  {
    path: '/',
    errorElement: <NotFound />,
    children: [
      { index: true, element: <Home /> },
      { path: 'ai-game', element: <AiGame /> },
      { path: 'strategy-lookup', element: <StrategyLookup /> },
      { path: 'auth/callback', element: <AuthCallback /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;

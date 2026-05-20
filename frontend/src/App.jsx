// C:\Ron\AllIn\frontend\src\App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import AiGame from './pages/AiGame';
import StrategyLookup from './pages/StrategyLookup';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/ai-game" element={<AiGame />} />
        <Route path="/strategy-lookup" element={<StrategyLookup />} />
      </Routes>
    </Router>
  );
}

export default App;


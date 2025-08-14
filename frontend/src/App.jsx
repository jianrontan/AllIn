// C:\Ron\AllIn\frontend\src\App.jsx
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import AiGame from './pages/AiGame';
import StrategyLookup from './pages/StrategyLookup';
import './styles.css';

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


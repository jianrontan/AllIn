# ALLIN: Play to Win

AllIn is a web-based poker learning and playing project designed to help users master poker strategy through interactive gameplay, pre-computed strategies, and future live strategy computation for additional accuracy. The system features a sophisticated Monte Carlo CFR poker bot integrated with a modern React frontend for strategy lookup and analysis.

---

## Milestone 2 Submission

### Level of Achievement: Apollo

- **Complete AI Engine Implementation:**  
  - Fully functional Monte Carlo Counterfactual Regret Minimization (CFR) poker bot
  - 1000+ unique information sets discovered through training
  - Sophisticated card abstractions (8 preflop + 6 postflop buckets) and action abstractions (pot-relative betting)
  - Integration with PyPokerEngine and phevaluator for realistic gameplay and hand evaluation
  - Convergent strategy learning with normalized mixed strategies

- **Interactive Frontend Application:**
  - React-based strategy lookup interface with real-time poker situation analysis
  - Intuitive card input system for hole cards and community cards
  - Dynamic game state builder with action history tracking and pot calculation
  - Strategy visualization with percentage breakdowns and weighted random action selection
  - Responsive design with modern UI components

- **Python Backend API Integration:**
  - Flask-based API bridge connecting frontend to poker bot abstractions
  - Real-time strategy lookup using exact training logic (no JavaScript approximations)
  - Comprehensive error handling and input validation
  - CORS-enabled for seamless frontend-backend communication

---

### Project Scope

**One-Sentence Scope:**  
A poker strategy learning platform featuring an AI poker bot trained with Monte Carlo CFR, integrated with a React frontend for real-time strategy lookup and analysis.

**Descriptive Scope:**  
For Milestone 2, the project has evolved into a comprehensive poker AI system consisting of three main components: (1) a Monte Carlo CFR poker bot capable of learning Game Theory Optimal strategies across 1000+ unique poker situations, (2) a React frontend providing intuitive strategy lookup with card input, game state building, and strategy visualization, and (3) a Python Flask API backend that bridges the frontend to the bot's exact abstraction logic. Users can input specific poker scenarios (hole cards, community cards, pot size, action history) and receive the AI's learned mixed strategies with percentage breakdowns. The system demonstrates sophisticated poker AI development while maintaining user-friendly interaction through modern web technologies.

---

## Key Features (Milestone 2)

### AI Engine & Strategy Generation
- **Monte Carlo CFR Implementation**: Core algorithm with 1,000+ iteration training capacity
- **Card Abstractions**: 8 preflop hand categories + 6 postflop strength levels (monster, strong, medium, weak_made, draw, bluff)
- **Action Abstractions**: Pot-relative bet sizing (33%, 66%, 100% pot) with realistic poker sequences
- **Strategy Convergence**: Normalized mixed strategies with mathematical validation
- **Information Set Discovery**: 1,238+ unique game situations automatically discovered during training

### Frontend Application
- **Strategy Lookup Interface**: Input specific poker situations and receive AI recommendations
- **Card Input System**: Intuitive hole card and community card entry with street selection
- **Game State Builder**: Dynamic action history creation with pot size and stack tracking
- **Strategy Visualization**: Mixed strategy display with percentage breakdowns and probability bars
- **Weighted Random Selection**: "Get Strategy" button that respects learned probability distributions

### Backend Integration
- **Python Flask API**: RESTful API connecting frontend to poker bot logic
- **Exact Abstraction Logic**: Uses identical card/action abstractions as training (no approximations)
- **Real-time Strategy Lookup**: Sub-second response times using pre-computed strategies
- **Comprehensive Error Handling**: Input validation and graceful error messaging

---

## Tech Stack

### AI & Backend
- **Python**: Core language for AI implementation
- **Monte Carlo CFR**: Game theory algorithm for strategy learning
- **PyPokerEngine**: Realistic poker game simulation and rules
- **phevaluator**: High-performance poker hand evaluation (C library)
- **Flask**: Lightweight web framework for API development
- **NumPy**: Numerical computations and strategy calculations

### Frontend
- **React (with Vite)**: Modern frontend framework with fast development server
- **JavaScript ES6+**: Component-based architecture with hooks
- **CSS3**: Responsive styling with flexbox and grid layouts
- **Fetch API**: HTTP client for backend communication

### Development & Testing
- **Git**: Version control with feature branch workflow
- **JSON**: Strategy storage and API communication format

---

## Milestone 2 Achievements

### Core AI Development
- ✅ Implemented complete Monte Carlo CFR algorithm for poker strategy learning
- ✅ Developed sophisticated card and action abstraction systems
- ✅ Achieved training convergence with 1,238+ discovered information sets
- ✅ Integrated with professional poker libraries (PyPokerEngine, phevaluator)

### User Interface Development  
- ✅ Built intuitive React-based strategy lookup interface
- ✅ Implemented dynamic game state building with action history
- ✅ Created real-time strategy visualization with percentage displays
- ✅ Added weighted random action selection based on learned probabilities

### System Integration
- ✅ Developed Python Flask API for frontend-backend communication
- ✅ Achieved seamless integration using exact training logic (no approximations)
- ✅ Established real-time strategy lookup with sub-second response times

---

## Known Issues & Next Steps

### Planned Enhancements Overview (Milestone 3)
- **Database Integration**: SupaBase for persistent hand history and user data storage
- **Live Gameplay**: Real-time poker game interface with bot vs human play
- **Subgame Solving**: Advanced real-time strategy refinement for critical decisions
- **Multiplayer Support**: WebSocket-based multiplayer poker rooms
- **Advanced Analytics**: Hand history analysis and strategy leak detection

### Current Limitations
- Strategy lookup only (no live gameplay yet)
- Limited to heads-up poker scenarios
- No persistent user data or hand history storage
- No opponent modeling or exploitative play

### To Do
Low priority:
- Multi-way support (not just heads up)

Medium priority:
- Sub game solving
- Monte Carlo CFR improvements
- Pruning
- Input restriction (frontend)

High priority:
- Support more legal actions
- Improving abstraction maybe adding stack sizes (need to research usefulness)
- Enforce poker rules (e.g. no raising < 2x previous bet)
- Fix project directory structure
- Fully connect bot to the frontend, not just the json file

---

## Getting Started

### Prerequisites
- Node.js (v16+)
- Python (3.8+)
- Git

### Frontend Setup
1. Clone the repository:
```
git clone https://github.com/jianrontan/AllIn.git
cd AllIn
```
2. Install frontend dependencies:
```
npm install
```
3. Start the React development server:
```
npm run dev
```

### Backend Setup
1. Install Python dependencies:
```
pip install flask flask-cors numpy phevaluator
```
2. Start the Flask API server:
```
cd backend
python strategy_api.py
```

### Using the Strategy Lookup
1. Open the frontend link from frontend terminal
2. Click "Look up Strategy" from the home page
3. Input your poker situation:
   - Enter hole cards (e.g., AS, AC)
   - Select street and add community cards if needed
   - Build action history using the dynamic interface
4. Click "Find Strategy" to get AI recommendations
5. Use "Get Random Strategy" for weighted action selection

## Training Your Own Strategies
```
python -m tests.test_blueprint_trainer
```
---

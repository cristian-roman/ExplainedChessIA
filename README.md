# ExplainedChessIA – Explainable Chess AI Engine with Natural Language Interface

This project implements an **Explainable AI (XAI)** module for chess. It uses a trained neural network to interpret and respond to chess-related natural language questions, integrating **stockfish** as a chess engine and exposing its capabilities via a **Flask-based web interface**.

---

## 🧠 Features

- Accepts natural language queries related to chess positions or sequences.
- Provides reasoning behind recommended moves using an explainable neural model.
- Offers integration with a browser-based graphical interface.
- Supports inference and training from structured Q&A data.

---

## 🗂️ Structure Overview

```
ExplainedChessIA-main/
├── ExplainedChess/             # AI backend and Flask server
│   ├── AI/                     # Contains explainability model code
│   ├── Server/                 # Flask API to serve model predictions
│   └── main.py                 # Launch server or train model
├── ExplainedChessInterface/    # HTML/CSS/JS interface for interaction
├── Tema1/                      # Homework project using Q&A for training
├── Tema2/                      # Homework project analyzing game sequences
├── all-data.txt                # Master file of training questions
├── chat.txt                    # Training data used in `Tema1`
└── README.md                   # Documentation
```

---

## 🧰 Requirements

Ensure **Python 3** is installed, then install the following libraries:

```bash
pip install torch stockfish python-chess numpy flask flask_cors
```

---

## ▶️ How to Run

1. **Start the Flask API Server**:

```bash
cd ExplainedChess
python3 main.py
```

2. **Open the Interface**:

Open `ExplainedChessInterface/interface.html` in a browser to interact with the AI.

---

## 💬 Supported Question Formats

You can ask a wide range of chess-related questions. However, the following two formats are specially supported for in-depth move recommendations:

### ✅ Format 1 – Sequence of Moves

**Example:**

```
Avand urmatoarea partida e4 e5 care sunt urmatoarele cele mai bune 2 mutari?
```

- Move sequence must be valid and space-separated.
- The number of moves to suggest should be between **1 and 9**.
- AI returns recommended next moves and **win probabilities**.

### ✅ Format 2 – FEN Position

**Example:**

```
Avand urmatoarea pozitie rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 care sunt cele mai bune 2 mutari in avans?
```

- Provide a valid **FEN** notation followed by a request for N best moves (1–9).
- AI will simulate and return move recommendations with estimated winning chances.

---

## ❌ Unsupported Question Prefixes

Avoid questions starting with:

- `Avand urmatoarea partida de sah`
- `Avand urmatoarea pozitie de sah`

…unless they exactly follow the correct formats described above.

---

## 🧪 Training Instructions

To retrain the neural network:

1. Open `ExplainedChess/main.py`
2. **Uncomment** the training-related code and **comment out** the server-starting code.
3. Use `chat.txt` as the training data, formatted as:

```
Q1
A1
Q2
A2
...
```

Questions and answers must be **separated only by a single space**.

4. Suggested approach:
   - Copy all questions from `all-data.txt` into `chat.txt`
   - Train until loss reaches approx. **0.001**

---

## 🎓 Educational Focus

- Neural Network Training on custom dialogue data
- Chess AI with embedded explainability
- Natural Language Understanding in Rule-Based Domains
- Flask + Web Interface for AI Integration


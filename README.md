# ♟️ ExplainedChessIA – Explainable Chess AI with Natural Language Interface

An **Explainable AI (XAI)** system that combines a neural language model with the **Stockfish** chess engine to answer natural language questions about chess positions and move sequences, while also providing reasoned explanations behind its recommendations.

The project focuses on interpretability, natural language interaction, and human–AI communication in a rule-based domain.

---

## 🧠 Project Overview

Unlike traditional chess engines that return only the best move, this system:
* **Accepts natural language questions** about chess.
* **Translates queries** into structured engine commands.
* **Uses a neural model** to generate human-readable explanations.
* **Exposes a web interface** for real-time interaction.

This makes it an ideal case study for **Explainable AI concepts**, **Decision Support Systems**, and **NLP in rule-heavy domains**.

---

## 🗂️ Repository Structure

```text
ExplainedChessIA/
├── ExplainedChess/
│   ├── AI/                     # Explainability & neural model logic
│   ├── Server/                 # Flask REST API
│   └── main.py                 # Train model or start server
├── ExplainedChessInterface/    # HTML/CSS/JS client interface
├── Tema1/                      # Q&A-based training experiments
├── Tema2/                      # Game-sequence analysis experiments
├── all-data.txt                # Consolidated training questions
├── chat.txt                    # Training dataset (Q/A pairs)
└── README.md
```

---

## 🔬 Core Features

* ♟ **Chess Reasoning:** High-level analysis via the Stockfish engine.
* 🧠 **Explanation Module:** PyTorch-based neural network for reasoning.
* 💬 **NL Interface:** Understands and processes natural language input.
* 🌐 **Full-Stack:** Flask API backend with a vanilla JS frontend.
* 📊 **Probabilistic Evaluation:** Win probability estimation for recommendations.
* 🔁 **Retrainable:** Easily fine-tune the model with custom Q&A datasets.

---

## 🛠️ Technologies Used

* **Language:** Python 3
* **Deep Learning:** PyTorch
* **Chess Logic:** Stockfish, `python-chess`
* **Backend:** Flask, Flask-CORS
* **Numerical Computing:** NumPy
* **Frontend:** Vanilla HTML / CSS / JavaScript

---

## ▶️ Running the Project

1. **Install Dependencies:**
   ```bash
   pip install torch stockfish python-chess numpy flask flask_cors
   ```

2. **Start the Backend Server:**
   ```bash
   cd ExplainedChess
   python3 main.py
   ```

3. **Open the Interface:**
   Launch `ExplainedChessInterface/interface.html` in any modern web browser.

---

## 💬 Supported Question Formats

The system supports general chess questions using two primary structured formats:

### Format 1 – Move Sequence
* **Example:** `Avand urmatoarea partida e4 e5 care sunt urmatoarele cele mai bune 2 mutari?`
* **Input:** A space-separated move sequence.
* **Output:** Recommended continuations and estimated winning probabilities.

### Format 2 – FEN Position
* **Example:** `Avand urmatoarea pozitie rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2 care sunt cele mai bune 2 mutari in avans?`
* **Input:** A valid FEN (Forsyth-Edwards Notation) string.
* **Output:** Ranked suggestions with engine evaluations.



---

## 🧪 Training the Model

To retrain the explainability network:
1. Open `ExplainedChess/main.py`.
2. Comment out the server startup logic and uncomment the training section.
3. Ensure `chat.txt` is formatted as alternating lines of **Question** and **Answer** separated by a single space.
4. **Target:** Aim for a training loss of approximately `0.001`.

---

## 🎯 Key Learnings

This project demonstrates:
* **Hybrid AI:** Integration of symbolic systems (Stockfish) with neural models.
* **Interpretability:** Practical application of Explainable AI (XAI) principles.
* **Domain-Specific NLP:** Applying natural language processing to highly structured environments.
* **End-to-End Engineering:** From model architecture to API and UI deployment.

---

## 🎓 Academic Context
* **Topic:** Explainable AI & NLP
* **Level:** Undergraduate coursework / applied research
* **Focus:** Interpretability over raw playing strength

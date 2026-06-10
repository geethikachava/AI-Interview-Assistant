# AI Interview Assistant
An end-to-end **AI-powered mock interview system** that conducts interviews in the
browser, transcribes spoken answers, verifies candidate identity from the webcam, and
produces an objective, multi-dimensional hireability assessment with a final hire / no-hire
decision.

The scoring design is **inspired by IEEE research on automated asynchronous video
interview (AVI) assessment** — in particular the **MAG-BERT-ARL** family of models.

> **Developed by Chava Sri Geethika.**

---

## Research Inspiration — MAG-BERT-ARL

**MAG-BERT-ARL** stands for **Multimodal Adaptation Gate BERT with Adversarial
Representation Learning**. It is a research direction in automated interview assessment that
combines two ideas:

| Component | Idea | What it contributes |
|-----------|------|---------------------|
| **MAG-BERT** (Multimodal Adaptation Gate + BERT) | Fuses the **verbal** (text), **vocal** (audio prosody) and **visual** (facial) modalities into a single language-model representation instead of scoring text alone. | A richer, human-like read of *how* a candidate answers, not just *what* they say. |
| **ARL** (Adversarial Representation Learning) | Trains an adversary to *remove* protected-attribute information (e.g. gender) from the learned representation. | Reduces demographic **bias**, pushing scores toward fairness and merit. |

This project is a **practical, lightweight reimagining** of those principles for a real,
deployable web application. Rather than training a heavy multimodal transformer, it
approximates the same multi-signal, fairness-aware philosophy with accessible components:

- **Verbal** signal → speech-to-text (Whisper) + LLM / keyword scoring of answer content.
- **Vocal / delivery** signal → communication, confidence and fluency heuristics
  (fillers, hedging, assertive phrasing, structure).
- **Visual** signal → webcam **face verification** to confirm the candidate's identity.
- **Fairness in spirit** → scoring rubrics key on *job-relevant* skills and work values,
  not on personal/demographic attributes.

---

## Features

-  **Authentication & roles** — separate flows for **candidates** and **interviewers**.
-  **Interviewer dashboard** — create interview sets, auto-generate questions with AI,
  and share a unique attend-link with candidates.
-  **Live browser interview** — records spoken answers question-by-question.
-  **Speech-to-text** — offline transcription with **faster-whisper** (CPU / int8).
-  **Face verification** — confirms the live candidate matches their profile photo using
  OpenCV (Haar-cascade detection + LBPH + colour-histogram matching).
-  **AI question generation** — role/difficulty-aware questions via the Groq LLM API,
  with a built-in fallback question bank.
-  **Multi-dimensional scoring** — communication, confidence, keywords matched, work
  values, and technical relevancy → weighted technical & soft scores → overall score.
-  **Hire decision** — Exceptional Hire / Strong Hire / Hire / Borderline / No Hire.
-  **History & improvement tracking** — stores past sessions and shows progress over time.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, Flask |
| **Frontend** | HTML (Jinja2 templates), CSS, JavaScript |
| **Database** | SQLite |
| **Speech-to-text** | faster-whisper |
| **Computer vision** | OpenCV (Haar cascade, LBPH, histogram matching) |
| **LLM (questions & scoring)** | Groq API — `llama-3.3-70b-versatile` |

---

## Project Structure

```
ai_interview_web/
├── src/
│   ├── backend/
│   │   ├── app.py             # Flask app, routes, session handling
│   │   ├── database.py        # SQLite schema + data access
│   │   ├── ai_questions.py    # AI question generation (Groq + fallback)
│   │   ├── web_scorer.py      # Multi-dimensional answer scoring & decision
│   │   ├── transcriber.py     # faster-whisper speech-to-text
│   │   ├── face_verify.py     # OpenCV identity verification
│   │   ├── test.py            # Tests / scratch
│   │   └── requirements.txt   # Python dependencies
│   └── frontend/              # Jinja2 HTML templates
│       ├── index.html         # Login / signup
│       ├── profile.html       # Candidate profile
│       ├── interviewer_*.html # Interviewer profile & dashboard
│       ├── interview.html     # Live interview UI
│       ├── results.html       # Score report
│       ├── history.html       # Past sessions & improvement
│       └── error.html
├── README.md
└── .gitignore
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier) for AI question generation and
  LLM scoring. *(Optional — the app falls back to keyword scoring and a fallback question
  bank without it.)*

### 2. Install

```bash
cd src/backend
pip install -r requirements.txt
```

> Face verification additionally needs OpenCV with the `face` module
> (`pip install opencv-contrib-python`).

### 3. Configure environment

Create a `.env` file inside `src/backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> The `.env` file is git-ignored and is **never** committed.

### 4. Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## How It Works

1. **Sign up / log in** as a candidate or an interviewer.
2. An **interviewer** creates an interview set (or auto-generates questions) and shares the
   attend-link.
3. A **candidate** completes their profile (including a profile photo for identity checks)
   and opens the link.
4. During the interview, the webcam **verifies identity** and the browser **records each
   answer**.
5. Audio is **transcribed** offline, then each answer is **scored** across five dimensions.
6. The system computes a weighted **overall score** and renders a **hire decision** with
   targeted improvement suggestions.
7. Results are saved to **history** so candidates can track improvement over time.

---

## Scoring Dimensions

| Dimension | What it measures |
|-----------|------------------|
| **Communication** | Structure, fluency, examples, low filler usage |
| **Confidence** | Assertive vs. hedging language |
| **Keywords Matched** | Coverage of expected technical concepts |
| **Technical Relevancy** | Depth and correctness of the technical content |
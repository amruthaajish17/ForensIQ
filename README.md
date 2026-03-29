# 🔍 ForensIQ — AI Crime Scene Analysis Assistant

> Forensic Intelligence, Powered by AI

ForensIQ is a multimodal AI system that analyzes crime scene images and generates
structured forensic investigation reports in real-time. It combines deep learning-based
object detection, retrieval-augmented generation (RAG) over real forensic knowledge,
and large language model reasoning into a single end-to-end pipeline.

---

## 🎯 Problem Statement

First responders and forensic investigators often operate under time pressure at crime
scenes with limited access to forensic expertise. ForensIQ acts as an AI co-investigator —
instantly analyzing a scene, surfacing relevant forensic protocols, and generating
actionable investigation plans.

---

## 🧠 System Architecture
```
Crime Scene Image
       │
       ▼
┌─────────────────────┐
│  YOLOv8 (Vision)    │  ← Object detection (deep learning)
│  + ForensicScorer   │  ← Priority scoring (Adam optimizer)
└────────┬────────────┘
         │ Structured Scene Text
         ▼
┌─────────────────────┐
│   FAISS RAG Layer   │  ← Cosine similarity search
│  Wikipedia Corpus   │  ← 200+ real forensic passages
└────────┬────────────┘
         │ Retrieved Forensic Context
         ▼
┌─────────────────────┐
│   Claude Sonnet     │  ← LLM reasoning + report generation
└────────┬────────────┘
         │
         ▼
  Investigation Report
  (Evidence · Actions · Tests · Scenario)
```

---

## ✨ Features

- 🖼️ **Multimodal Input** — Upload any crime scene image (JPG/PNG)
- 👁️ **Deep Learning Vision** — YOLOv8 detects objects with confidence scores
- 📚 **Real Forensic Knowledge** — Wikipedia corpus across 15 forensic science domains
- ⚡ **FAISS Vector Search** — Fast cosine similarity retrieval over 200+ passages
- 🧠 **LLM Reasoning** — Claude generates structured, actionable investigation reports
- 📉 **Adam Optimizer** — ForensicScorer neural network trained live with loss visualization
- 🎛️ **Interactive Dashboard** — Streamlit UI with real-time analysis and expandable sources

---

## 🧩 Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning Framework | PyTorch |
| Optimizer | Adam (lr=0.001, β₁=0.9, β₂=0.999) + StepLR Scheduler |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS (IndexFlatIP, cosine similarity) |
| Knowledge Source | Wikipedia API (15 forensic topics) |
| LLM | Claude Sonnet (Anthropic API) |
| Frontend | Streamlit |

---

## 📁 Project Structure
```
forensiq/
├── app.py              # Streamlit UI — main entry point
├── vision.py           # YOLOv8 inference + ForensicScorer + Adam training
├── rag.py              # Wikipedia corpus loader + FAISS index + retrieval
├── llm.py              # Claude API integration + report generation
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/forensiq.git
cd forensiq
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key
```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (CMD)
set ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Dependencies
```txt
ultralytics
torch
torchvision
streamlit
faiss-cpu
sentence-transformers
anthropic
wikipedia-api
matplotlib
Pillow
numpy
```

---

## 🔬 How It Works

### Step 1 — Vision Analysis
YOLOv8, a state-of-the-art object detection model, scans the uploaded image and
identifies all visible objects with confidence scores. Detections are filtered
(>0.3 confidence) and converted into a structured scene description.

### Step 2 — Forensic Scoring (Adam Optimizer)
A lightweight neural network (`ForensicScorer`) assigns forensic priority scores
to detected objects. It is trained in real-time using the **Adam optimizer**
with the following configuration:
```python
optimizer = torch.optim.Adam(
    scorer.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)
```

A StepLR scheduler halves the learning rate every 10 epochs. The loss curve
is displayed live on the dashboard.

### Step 3 — RAG Retrieval
The scene description is embedded using `sentence-transformers` and queried
against a FAISS index built from 200+ forensic science paragraphs sourced
from Wikipedia across 15 domains including blood spatter analysis, DNA profiling,
digital forensics, chain of custody, and Locard's exchange principle.

### Step 4 — LLM Reasoning
The scene description and retrieved passages are passed to Claude Sonnet,
which generates a structured investigation report covering:

- Key evidence identified
- Immediate actions (first 30 minutes)
- Recommended forensic tests
- Investigation priority score (1–10)
- Possible scenario reconstruction

---

## 📊 Optimization Methods

| Method | Purpose |
|---|---|
| **Adam Optimizer** | Training ForensicScorer neural network |
| **StepLR Scheduler** | Adaptive learning rate decay |
| **FAISS ANN Search** | Approximate nearest-neighbor retrieval |
| **L2 Normalization** | Cosine similarity in vector space |
| **Dropout (0.2)** | Regularization in ForensicScorer |

---

## 🎓 Academic Context

This project was developed as a capstone for an advanced optimization and deep
learning course. It demonstrates the integration of:

- Module 1–2: Constraint formulation and nonlinear modeling
- Module 3: Real-time pipeline scheduling
- Module 4: Graph-based retrieval (FAISS)
- Module 5: Gradient optimization (Adam) and machine learning

---

## ⚠️ Disclaimer

ForensIQ is an academic prototype built for educational purposes. It is not
intended for use in real criminal investigations. All analysis should be
reviewed by qualified forensic professionals.

---

## 👥 Team

Built in 2 hours as part of a capstone project.

| Role | Responsibility |
|---|---|
| Amrutha Ajish Achuthan | Vision pipeline (YOLOv8, ForensicScorer, Adam training) |
| Akhila Sunesh | RAG system (Wikipedia, FAISS), LLM integration, Streamlit UI |

---

## 📄 License

MIT License — free to use, modify, and distribute for educational purposes.

# AI-Invoice-Processor (RAG System)

[![Live App](https://img.shields.io/badge/Live%20Demo-Render-blue?style=for-the-badge&logo=render)](https://ai-invoice-processor-rag.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

An automated, intelligent invoice parsing and query-answering system. Built using **Retrieval-Augmented Generation (RAG)**, **LangChain**, and **LangGraph** with a premium, light-glassmorphic web dashboard.

🔗 **Deployed Web URL:** [https://ai-invoice-processor-rag.onrender.com](https://ai-invoice-processor-rag.onrender.com)

---

## 🌟 Key Features

* **Smart PDF Parsing & Chunking**: Extracts text from unstructured PDF invoices and splits it dynamically into overlapping semantic chunks.
* **Agentic State Flow (LangGraph)**: Orchestrates the pipeline state machine through standard nodes (`START` ➔ `load_pdf` ➔ `retrieve` ➔ `generate` ➔ `END`).
* **Hybrid Search Retrieval**: 
  * **Online Mode**: Integrates with Google Generative AI Embeddings and Vector Search.
  * **Offline Mode (Zero-Friction Fallback)**: Automatically falls back to a local BM25 ranking algorithm and regex heuristic rule extractor if offline or if no API key is present.
* **Premium Glassmorphic Dashboard**: 
  * Interactive chat assistant window.
  * Real-time LangGraph execution visualizer showing node status updates dynamically.
  * Overlay modal dialogs to inspect raw document chunks and retrieved context blocks.
  * Integrated instructions guide directly inside the application header.

---

## 🛠️ Technology Stack

* **Backend**: FastAPI, Uvicorn, Python
* **AI & Orchestration**: LangGraph, LangChain, Google Gemini API (`gemini-2.5-flash`)
* **Retrieval & Parsing**: PyPDF, Rank-BM25
* **Frontend**: HTML5, Vanilla CSS (Glassmorphism, custom light/dark theme toggles), JavaScript

---

## 🚀 Local Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/SRINI-SEENI/AI-Invoice-Processor-RAG.git
cd AI-Invoice-Processor-RAG
```

### 2. Configure Virtual Environment
Create and activate a local Python virtual environment:

**On Windows (PowerShell)**:
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

**On macOS/Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 5. Launch the Server
Start the local FastAPI development server:
```bash
python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser to run the interface.

### 6. Run Automated Tests
Execute the testing suite containing pre-configured validation queries:
```bash
python test_rag.py
```

---

## 🌐 Production Deployment

The project is pre-configured for seamless deployment to platforms like **Render**:

* **Build Command**: `pip install -r requirements.txt`
* **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
* **Health Check Path**: `/`
* **Environment Variable**: Add `GEMINI_API_KEY` under the Environment Settings.

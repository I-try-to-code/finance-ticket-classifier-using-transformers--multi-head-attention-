# Banking77 Intent Classifier &mdash; Transformer Inference System

A production-grade, end-to-end NLP system for banking customer support intent classification across **77 fine-grained categories** from the [Banking77](https://huggingface.co/datasets/PolyAI/banking77) benchmark.

Powered by a fine-tuned **DistilBERT** model yielding **0.8907 Validation Macro F1**, backed by a high-performance **FastAPI** inference microservice (~18–20 ms GPU latency), and served with a responsive **Vanilla Web UI**.

---

## Key Highlights & Architecture

- **Selected Production Model**: `distilbert-base-uncased` fine-tuned end-to-end on 77 intent classes.
- **Controlled 3-Way Benchmark**: Rigorously validated against TF-IDF Baseline V1 and BERT-base V3. DistilBERT was selected for production due to achieving 99.7% of BERT's F1 score with **2.6x lower inference latency** and half the parameter footprint.
- **FastAPI Backend**: Lifespan startup handler for single-instance model loading, CUDA kernel pre-warming, inference-only softmax, and `torch.no_grad()` optimization.
- **Dual-Mode Frontend**: Clean, responsive, glassmorphic dark-theme UI with live health telemetry, interactive example queries, and top-3 probability visualizations. Can be served directly by FastAPI or hosted standalone.

---

## 3-Way Model Benchmark Comparison

All experiments used the exact same stratified data split (Train: 8,002 | Validation: 2,001 | Test: 3,080 locked).

| Metric / Attribute | Baseline V1 (TF-IDF + LogReg) | DistilBERT V2 (Selected) | BERT-Base V3 |
| :--- | :--- | :--- | :--- |
| **Architecture** | Word + Char N-Grams + Logistic Regression | `distilbert-base-uncased` (6 layers, 12 heads) | `bert-base-uncased` (12 layers, 12 heads) |
| **Parameter Count** | ~35,000 coefficients | **66.4 Million** | 109.5 Million |
| **Validation Macro F1** | 0.8143 | **0.8907** (+0.0764 vs V1) | 0.8931 (+0.0024 vs V2) |
| **Validation Weighted F1** | 0.8148 | **0.9179** | 0.9197 |
| **Validation Accuracy** | 81.56% | **91.85%** | 92.00% |
| **GPU Inference Latency** | N/A (CPU only) | **~18 &ndash; 20 ms / query** | ~48 &ndash; 52 ms / query |
| **Model Disk Size** | 2.1 MB | **268 MB** (`model.safetensors`) | 438 MB |
| **Decision Rationale** | Strong linear baseline | **Production Selection** &mdash; optimal latency/accuracy tradeoff | Diminishing returns (+0.24% F1 for 2.6x inference cost) |

---

## System Architecture

```mermaid
graph LR
    User[User / Support Agent] -->|Browser / HTTP| UI[Frontend UI (Vanilla HTML/CSS/JS)]
    UI -->|POST /predict| API[FastAPI Inference Service]
    API -->|Batch/Single Query| Predictor[Banking77Predictor]
    Predictor -->|Tokenization| Tokenizer[DistilBERT Tokenizer]
    Predictor -->|Forward Pass| Model[DistilBERT V2 Checkpoint]
    Model -->|Logits -> Softmax| Output[Top-K Probabilities & Latency]
    Output --> API
    API --> UI
```

---

## Project Directory Structure

```text
├── data/
│   ├── raw/                      # Banking77 raw datasets (train.csv, test.csv)
│   └── processed/                # Stratified train (8,002) and validation (2,001) splits
├── models/
│   ├── distilbert_banking77/     # Best DistilBERT checkpoint (safetensors, tokenizer, config)
│   └── bert_banking77/           # Best BERT-base checkpoint
├── reports/                      # Error analysis reports and confusion matrices
├── frontend/
│   ├── index.html                # Semantic HTML5 application interface
│   ├── style.css                 # Dark fintech theme (glassmorphism, CSS grid, variables)
│   ├── config.js                 # API endpoint resolution & local configuration
│   └── app.js                    # Client application logic, validation & rendering
├── src/
│   ├── api/
│   │   ├── app.py                # FastAPI application, CORS, and static file hosting
│   │   ├── config.py             # API configuration via Pydantic settings
│   │   ├── schemas.py            # Request/response validation contracts
│   │   └── main.py               # Uvicorn entrypoint
│   ├── inference/
│   │   └── predictor.py          # Standalone Banking77Predictor component
│   ├── distilbert/               # DistilBERT training and dataset modules
│   └── bert/                     # BERT training and dataset modules
├── tests/
│   ├── test_inference.py         # Unit tests for Banking77Predictor
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── run_inference_demo.py         # CLI inference runner
├── requirements.txt              # Production and development dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ and a virtual environment set up:

```bash
# Clone the repository
git clone https://github.com/I-try-to-code/finance-ticket-classifier-using-transformers--multi-head-attention-.git
cd "finance ticket classifier using transformers (multi head)"

# Install dependencies
pip install -r requirements.txt
```

### 2. Starting the Backend Server

Start the FastAPI application via the built-in runner or Uvicorn:

```bash
# Using the module runner
python -m src.api.main --host 0.0.0.0 --port 8000

# Or directly with Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Upon startup, the server initializes the predictor, warms up the PyTorch CUDA kernels (if a GPU is detected), and binds to `http://localhost:8000`.

- **Web UI**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive OpenAPI Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Service Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Running the Frontend

### Mode A: Hosted directly by FastAPI (Recommended)
When the FastAPI server starts, it automatically mounts `frontend/` at the root path:
- Open your browser to **`http://localhost:8000`**.
- No additional web server configuration is required.

### Mode B: Standalone Web Server
If you prefer running the frontend in isolation (e.g. via a CDN or local static server):

```bash
# Start a lightweight Python HTTP server in the frontend directory
python -m http.server 3000 --directory frontend
```
- Open **`http://localhost:3000`** in your browser.
- The UI automatically points to `http://localhost:8000` or can be reconfigured dynamically via the gear icon (**Settings**) in the top navigation bar.

---

## API Reference

### 1. Health Check
`GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "model_name": "distilbert-base-uncased (Banking77 V2)",
  "device": "cuda:0",
  "model_loaded": true
}
```

### 2. Intent Prediction
`POST /predict`

**Request Body**:
```json
{
  "text": "My transfer is still pending and has not arrived"
}
```

**Response (`200 OK`)**:
```json
{
  "text": "My transfer is still pending and has not arrived",
  "predicted_intent": "pending_transfer",
  "confidence": 0.9632,
  "top_predictions": [
    {
      "intent": "pending_transfer",
      "probability": 0.9632
    },
    {
      "intent": "transfer_not_received_by_recipient",
      "probability": 0.0154
    },
    {
      "intent": "balance_not_updated_after_cheque_or_cash_deposit",
      "probability": 0.0041
    }
  ],
  "latency_ms": 18.84
}
```

---

## Running Tests & CLI Demo

### Run CLI Prediction Demo
Test 5 representative banking queries directly from the command line:

```bash
python run_inference_demo.py
```

### Run Unit & Integration Tests
Execute test suites covering predictor inference and FastAPI endpoints:

```bash
# Run predictor tests
pytest tests/test_inference.py -v

# Run FastAPI endpoint tests
pytest tests/test_api.py -v
```

---

## License & Citation
Developed for banking intent classification research using Hugging Face Transformers and the Banking77 benchmark.

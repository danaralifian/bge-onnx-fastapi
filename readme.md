A minimal, fast, and self-hosted API for generating text embeddings using the `bge-small-en` model from BAAI, converted to ONNX for optimized inference. Designed to run smoothly on devices with low RAM (1GB) or small VPS.

### 🌟 Features

- Built with FastAPI + Uvicorn (1 worker, lightweight)
- ONNX inference (no PyTorch in runtime)
- JSON API: `POST /embed` with batch support
- Easy to run on local or 1GB VPS (e.g. Oracle, DigitalOcean, etc.)

---

### 📦 How to Run (Locally)

1. 🔧 Install Python & Create Virtual Environment

```bash
# Make sure Python 3.10+ is installed
python --version

# Create & activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

```

2. 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

3. 📦 Download and Convert Model to ONNX

```bash
python convert_to_onnx.py
```

This will:

- Download BAAI/bge-small-en model
- Export the model to ONNX format (onnx-model/)
- Save compatible tokenizer to the same folder

4. 🚀 Run the API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Server will be available at http://localhost:8000.

Example Request

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Artificial Intelligence is transforming industries.", "I love pizza."]}'
```

Returns:

```bash
{
  "embeddings": [[...], [...]]
}
```

## Authors

- [@danaralifian](https://www.linkedin.com/in/danar-alifian-1a1581174/)

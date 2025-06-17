# Stage 1: build and convert ONNX
FROM python:3.10-slim AS builder
WORKDIR /build

COPY requirements-build.txt .
RUN pip install --no-cache-dir -r requirements-build.txt

COPY convert_to_onnx.py .
RUN python convert_to_onnx.py

# Stage 2: minimal runtime
FROM python:3.10-slim
WORKDIR /app

# Copy ONNX model & tokenizer files
COPY --from=builder /build/onnx-model /app/onnx-model

# Copy inference app and runtime dependencies
COPY app.py .
COPY requirements-runtime.txt .

# Install only lightweight runtime dependencies
RUN pip install --no-cache-dir -r requirements-runtime.txt

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

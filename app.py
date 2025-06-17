from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from fastapi.exceptions import RequestValidationError
from exceptions.handlers import (
    http_exception_handler,
    global_exception_handler,
    validation_exception_handler
)
from middlewares.response_formatter import ResponseFormatterMiddleware

app = FastAPI()

# Add middleware to globally format API responses
app.add_middleware(ResponseFormatterMiddleware)

# Register global exception handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("onnx-model")
session = ort.InferenceSession("onnx-model/model.onnx")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
def embed_text(data: TextInput):
    # Tokenize the input text
    inputs = tokenizer(data.text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prepare ONNX runtime input
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    # Run inference
    outputs = session.run(["last_hidden_state"], ort_inputs)
    embedding = outputs[0].mean(axis=1)[0]  # Apply average pooling

    # Return embedding vector (formatted by middleware)
    return {
        "text": embedding.tolist()
    }


@app.get("/status")
def healthcheck():
    try:
        # Example of checking ONNX session or load tokenizer
        if session is None:
            raise ValueError("ONNX session not ready")

        return {"status": "ok", "message": "ONNX model and tokenizer are ready"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/health-check")
def healthcheck():
    return {"status": "ok"}



from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from fastapi.responses import JSONResponse

app = FastAPI()

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("onnx-model")
session = ort.InferenceSession("onnx-model/model.onnx")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
def embed_text(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="np", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    outputs = session.run(["last_hidden_state"], ort_inputs)
    embedding = outputs[0].mean(axis=1)[0]  # Average pooling

    return {
        "embedding": embedding.tolist()
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
    return JSONResponse(content={"status": "ok"})



from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

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

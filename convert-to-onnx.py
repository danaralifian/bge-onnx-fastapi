# Convert model to onnx
from transformers import AutoTokenizer, AutoModel
import torch
import os

model_id = "BAAI/bge-small-en"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

# Save the tokenizer to the onnx-model folder
os.makedirs("onnx-model", exist_ok=True)
tokenizer.save_pretrained("onnx-model")

# Dummy input for tokenizer
inputs = tokenizer("Example input", return_tensors="pt")

# Convert to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "onnx-model/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=17
)

print("✅ model converted to onnx")

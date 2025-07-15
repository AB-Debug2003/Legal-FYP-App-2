from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

# Path to your extracted model directory
model_path = "./fine_tuned_pegasus_model"

# Load model & tokenizer
tokenizer = PegasusTokenizer.from_pretrained(model_path)
model = PegasusForConditionalGeneration.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test input
legal_text = """
The appellant challenged the decision of the High Court on the grounds that the evidence was inadmissible and the lower court erred in applying precedent.
"""

# Tokenize
inputs = tokenizer(legal_text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=512,
    num_beams=4,
    early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("🔍 Summary:\n", summary)

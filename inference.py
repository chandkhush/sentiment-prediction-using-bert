import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./bert_sentiment_model"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    if pred == 0:
        return "Negative 😞"
    else:
        return "Positive 😀"

# Example predictions

if __name__ == "__main__":
    while True:
        text = input("Enter text: ")

        if text.lower() in ["exit", "quit", "stop"]:
            break

        result = predict_sentiment(text)
        print("Prediction:", result)

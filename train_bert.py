import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load cleaned dataset
df = pd.read_csv("cleaned_tweets.csv", encoding="latin1", engine="python")

df = df.dropna(subset=["clean_text"])

# 50k training dataset
df = df.sample(50000, random_state=42)  

#  Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Load BERT tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization function
def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_ds = dataset.map(tokenize, batched=True, batch_size=500)

# Fix labels for Trainer
tokenized_ds = tokenized_ds.rename_column("target", "labels")

# Makes sure format is PyTorch tensors
tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load BERT for classification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # binary classification
)

# Training arguments (laptop safe)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,       
    gradient_accumulation_steps=4,       
    logging_steps=200,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save model

trainer.save_model("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")

print("Training completed! Model saved in ./bert_sentiment_model")

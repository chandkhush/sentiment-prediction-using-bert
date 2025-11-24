import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 1) Load cleaned dataset (small batches, safe for laptop)
df = pd.read_csv("cleaned_tweets.csv")
df = df.dropna(subset=["clean_text"])

dataset = Dataset.from_pandas(df)   # much safer than manual tensor creation

# 2) Load tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3) Tokenization function (runs lazily, not all at once)
def tokenize(batch):
    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_ds = dataset.map(tokenize, batched=True, batch_size=500)

# 4) Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 5) Training arguments (safe for small GPU/CPU)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,   # small batch = safe
    gradient_accumulation_steps=4,   # effective batch = 32
    logging_steps=200,
    save_strategy="epoch"
)

# 6) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
    tokenizer=tokenizer
)

# 7) Train
trainer.train()

# 8) Save model
trainer.save_model("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")

print("Training Completed!")

import pandas as pd
import re
import emoji
import contractions

#  Load dataset
file_path = "Sentiment140/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(file_path, encoding="latin-1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

#  Keep only text and target
df = df[["text", "target"]]
df["target"] = df["target"].replace({4:1})  # 0 = negative, 1 = positive

#  Define cleaning functions
def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#', '', text)
    text = contractions.fix(text)
    text = convert_emojis(text)
    text = normalize_repeated_chars(text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#  Apply cleaning function to create clean_text column
df["clean_text"] = df["text"].apply(clean_tweet)

#  Print first 5 rows — include clean_text column
print(df[["text","clean_text","target"]].head())

print(df.columns)

# Save cleaned dataset to CSV for training
df.to_csv("cleaned_tweets.csv", index=False)
print("Cleaned dataset saved as cleaned_tweets.csv")

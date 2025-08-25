import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting data preparation process...")

# Create directories
data_dir = Path('data')
test_dir = data_dir / 'testing_mail'
for directory in [data_dir, test_dir]:
    directory.mkdir(exist_ok=True)

# Read and preprocess data
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Log dataset statistics
spam_count = df['label'].sum()
logger.info(f"Dataset loaded: {len(df)} total messages ({spam_count} spam, {len(df)-spam_count} ham)")

# Split and save data
train_data = df.head(5246)
test = df.tail(300).copy()
test['mail_id'] = [f'test{i}.txt' for i in range(300)]

# Save training data
train_data.to_csv(data_dir / "train_data.csv", index=False)

# Create test files
logger.info("Creating test files...")
for i in range(300):
    file_path = test_dir / f"test{i}.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{test['message'].iloc[i]}\n")

# Save test targets
test[['mail_id', 'label']].to_csv(data_dir / 'test_targets.csv', index=False)

logger.info("Data preparation completed successfully")
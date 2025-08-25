# Email Spam Classifier

A custom implementation of a Naive Bayes classifier for email spam detection, built from scratch in Python.

## Overview
This project implements a simple but effective email spam classifier using:
- Custom CountVectorizer for text feature extraction
- Naive Bayes classifier with Laplace smoothing
- Logging for process tracking
- Custom accuracy calculation

## Prerequisites
- Python 3.8+
- pandas
- numpy
- Required dataset: `spam.csv` with columns 'v1' (label) and 'v2' (message)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/email_spam_classifier.git
cd email_spam_classifier
```

2. Install required packages:
```bash
pip install pandas numpy
```

## Project Structure
```
email_spam_classifier/
│
├── custom_class/
│   ├── __init__.py
│   ├── countvectorizer.py
│   └── classifier.py
│
├── data/
│   ├── train_data.csv
│   ├── test_targets.csv
│   └── testing_mail/
│
├── custom_train_test_split.py
├── train_test.py
└── README.md
```

## Usage

### Step 1: Prepare the Data
Run the data preparation script to split the dataset and create test files:
```bash
python custom_train_test_split.py
```
This will:
- Split the spam.csv dataset into train and test sets
- Create test email files in data/testing_mail/
- Save train_data.csv and test_targets.csv

### Step 2: Train and Test the Model
Run the training and testing script:
```bash
python train_test.py
```
This will:
- Load and vectorize the training data
- Train the Naive Bayes classifier
- Classify test emails
- Calculate and display accuracy
- Save results to final_results.csv

## Output Files
- `data/train_data.csv`: Training dataset
- `data/test_targets.csv`: True labels for test data
- `data/testing_mail/*.txt`: Individual test email files
- `final_results.csv`: Classification results with predictions

## Performance
The classifier typically achieves accuracy between 85-95% on the test set, depending on the data split.

## License
MIT License

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
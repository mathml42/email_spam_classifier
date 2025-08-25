from custom_class.countvectorizer import SimpleCountVectorizer
from custom_class.classifier import NaiveBayesClassifier
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting spam classification process...")

# Load training data
train_data = pd.read_csv("train_data.csv")
test_folder = 'testing_mail'

# Preprocess and vectorize training data
vectorizer = SimpleCountVectorizer()
X_train = vectorizer.fit_transform(train_data['message'])
y_train = train_data['label'].values

# Train the Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.train(X_train, y_train)
logging.info("Model training completed")

# Read test emails and classify
results = {}
file_count = 0
for filename in os.listdir(test_folder):
    if filename.endswith(".txt"):
        file_count += 1
        with open(os.path.join(test_folder, filename), 'r', encoding='utf-8') as file:
            email_content = file.read()
            X_test = vectorizer.transform([email_content])
            prediction = nb_classifier.predict(X_test)[0]
            results[filename] = "spam" if prediction == 1 else "ham"
logging.info(f"Classified {file_count} test emails")

# Create predictions DataFrame and calculate accuracy
predictions_df = pd.DataFrame({'mail_id': list(results.keys()), 
                             'predict': list(results.values())})
test_targets = pd.read_csv('test_targets.csv')
merged_df = predictions_df.merge(test_targets, on='mail_id')
merged_df['original'] = merged_df['label'].map({0: 'ham', 1: 'spam'})
accuracy = nb_classifier.calculate_accuracy(merged_df['original'], merged_df['predict'])
logging.info(f'Classification accuracy: {accuracy}%')

# Save results
merged_df.to_csv("final_results.csv", index=False)
logging.info("Classification process completed")
import numpy as np

class NaiveBayesClassifier:
    """
    A custom implementation of Naive Bayes Classifier for text classification.
    
    This classifier implements the Naive Bayes algorithm from scratch for binary
    classification (spam/ham). It uses log probabilities to prevent numerical
    underflow and implements Laplace smoothing.

    Attributes:
        word_probs (dict): Dictionary storing word probabilities for each class
        class_probs (dict): Dictionary storing prior probabilities of each class
    """

    def __init__(self):
        """
        Initialize the Naive Bayes classifier.

        Initializes empty dictionaries for storing:
        - Word probabilities for each class
        - Prior probabilities of each class
        """
        self.word_probs = {}  # Stores P(word|class) for each class
        self.class_probs = {} # Stores P(class)

    def train(self, X, y):
        """
        Train the Naive Bayes classifier using the provided data.
        
        Args:
            X (numpy.ndarray): Document-term matrix of shape (n_samples, n_features)
                             where each row represents a document and each column
                             represents a term from the vocabulary
            y (numpy.ndarray): Target labels of shape (n_samples,) with binary
                             values (0 for ham, 1 for spam)

        Returns:
            None
            
        Note:
            Uses Laplace smoothing (+1) to handle zero probabilities
            Stores log probabilities to prevent numerical underflow
        """
        n_samples, n_features = X.shape
        
        # Calculate class prior probabilities
        n_spam = np.sum(y == 1)
        n_ham = np.sum(y == 0)
        self.class_probs[1] = n_spam / n_samples  # P(spam)
        self.class_probs[0] = n_ham / n_samples   # P(ham)

        # Calculate conditional probabilities with Laplace smoothing
        spam_counts = X[y == 1].sum(axis=0) + 1  # Add 1 for Laplace smoothing
        ham_counts = X[y == 0].sum(axis=0) + 1   # Add 1 for Laplace smoothing
        total_spam = spam_counts.sum()
        total_ham = ham_counts.sum()

        # Store log probabilities
        self.word_probs[1] = np.log(spam_counts / total_spam)  # log P(word|spam)
        self.word_probs[0] = np.log(ham_counts / total_ham)    # log P(word|ham)

    def predict(self, X):
        """
        Predict class labels for input samples.

        Args:
            X (numpy.ndarray): Document-term matrix of shape (n_samples, n_features)
                             to predict labels for

        Returns:
            numpy.ndarray: Predicted class labels (0 for ham, 1 for spam)
            
        Note:
            Uses log probabilities for numerical stability
        """
        predictions = []
        for x in X:
            # Calculate log probability scores for each class
            spam_score = np.sum(x@(self.word_probs[1])) + np.log(self.class_probs[1])
            ham_score = np.sum(x@(self.word_probs[0])) + np.log(self.class_probs[0])
            # Predict spam if spam score is higher
            predictions.append(1 if spam_score > ham_score else 0)
        return np.array(predictions)
    
    def calculate_accuracy(self, true_labels, predictions):
        """
        Calculate classification accuracy.
        
        Args:
            true_labels (list/array): Ground truth labels
            predictions (list/array): Predicted labels
            
        Returns:
            float: Classification accuracy as a percentage
            
        Note:
            Accuracy = (correct predictions / total predictions) * 100
        """
        correct_predictions = 0
        for i in range(len(true_labels)):
            if true_labels[i] == predictions[i]:
                correct_predictions += 1
        
        accuracy = (correct_predictions / len(true_labels)) * 100
        return accuracy
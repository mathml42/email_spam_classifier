import numpy as np
import re

class SimpleCountVectorizer:
    """
    A simple implementation of CountVectorizer for text feature extraction.
    
    This class converts a collection of text documents to a matrix of token counts.
    It performs basic text preprocessing and builds a vocabulary of known words.
    """

    def __init__(self):
        """
        Initialize the SimpleCountVectorizer.
        
        Attributes:
            vocabulary_ (dict): A dictionary mapping words to their indices in the feature matrix
        """
        self.vocabulary_ = {}
    
    def fit(self, documents):
        """
        Learn the vocabulary from a list of documents.
        
        Args:
            documents (list): List of text documents to learn vocabulary from
            
        Returns:
            None
        """
        vocabulary = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            vocabulary.update(tokens)
        
        # Create word-to-index mapping
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocabulary))}
    
    def transform(self, documents):
        """
        Transform documents to a document-term matrix.
        
        Args:
            documents (list): List of text documents to transform
            
        Returns:
            numpy.ndarray: Document-term matrix where each row represents 
                          a document and each column represents a term from the vocabulary
        """
        rows = []
        for doc in documents:
            # Initialize count vector for document
            count_vector = [0] * len(self.vocabulary_)
            tokens = self._tokenize(doc)
            
            # Count token occurrences
            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    count_vector[index] += 1
            rows.append(count_vector)
        return np.array(rows)
    
    def fit_transform(self, documents):
        """
        Learn the vocabulary and transform documents to document-term matrix.
        
        Args:
            documents (list): List of text documents
            
        Returns:
            numpy.ndarray: Document-term matrix
        """
        self.fit(documents)
        return self.transform(documents)
    
    def _tokenize(self, text):
        """
        Convert text to tokens.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of tokens
            
        Note:
            - Converts text to lowercase
            - Removes special characters
            - Removes extra whitespace
        """
        # Remove special characters and convert to lowercase
        text = re.sub(r'\W+', ' ', text.lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower())
        return text.split()
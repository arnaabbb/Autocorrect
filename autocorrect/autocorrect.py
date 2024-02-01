import pandas as pd
import numpy as np

import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Get a sample text (you can choose a different corpus)
raw_text = gutenberg.raw('shakespeare-hamlet.txt')
# print(raw_text)

# Tokenize the text into words
words = word_tokenize(raw_text)

# Convert to lowercase
words = [word.lower() for word in words]

# Remove punctuation and non-alphabetic characters
words = [word for word in words if word.isalpha()]

# Remove stopwords
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Perform stemming (reduce words to their base or root form)
ps = PorterStemmer()
words = [ps.stem(word) for word in words]

# Print the preprocessed text
print(words[:50])  # Print the first 50 preprocessed words as an example


def edit_distance(str1, str2):
    """
    Compute the Levenshtein distance between two strings.
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],        # Deletion
                                   dp[i][j - 1],        # Insertion
                                   dp[i - 1][j - 1])    # Substitution

    return dp[m][n]

def get_closest_match(word, dictionary):
    """
    Get the closest match in the dictionary based on edit distance.
    """
    min_distance = float('inf')
    closest_match = None

    for candidate in dictionary:
        distance = edit_distance(word, candidate)
        if distance < min_distance:
            min_distance = distance
            closest_match = candidate

    return closest_match

# Example usage
correct_words = words
misspelled_word = input("Enter a word: ")
suggested_correction = get_closest_match(misspelled_word, correct_words)

print(f"Original word: {misspelled_word}")
print(f"Suggested correction: {suggested_correction}")

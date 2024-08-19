# NLP Co-Occurrence Matrix

## Overview

This project demonstrates natural language processing (NLP) techniques, including generating co-occurrence matrices, tokenization, and applying TF-IDF (Term Frequency-Inverse Document Frequency) representation. The script processes sample text to analyze word co-occurrences and compute cosine similarity between words and documents.

## Features

- **Bag of Words (BoW)**: Transform text data into a BoW matrix to analyze word co-occurrences.
- **Co-Occurrence Matrix**: Create a word co-occurrence matrix based on a sliding window approach.
- **Cosine Similarity**: Calculate cosine similarity between words and documents to measure their similarity.
- **TF-IDF Representation**: Convert text data into TF-IDF vectors for feature extraction and analysis.

## Requirements

- Python 3.x
- Required Python packages:
  - pandas
  - numpy
  - nltk
  - scikit-learn

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/nlp-co-occurrence-matrix.git
   cd nlp-co-occurrence-matrix
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that NLTK resources are downloaded by running the script or manually downloading them.

## Usage

1. **Bag of Words and Co-Occurrence Matrix**: The script creates a BoW matrix and computes a co-occurrence matrix based on a sliding window approach.

2. **Tokenization and Cleaning**: Tokenize and clean sample text by removing special characters.

3. **TF-IDF Representation**: Generate TF-IDF vectors from a set of text documents and compute their cosine similarity.

To run the script, use the following command:

```bash
python NLP_co_occurrence_matrix.py
```

## License

This project is licensed under the MIT License.

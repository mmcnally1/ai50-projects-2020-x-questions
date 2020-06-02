import nltk
import sys
import os
import string
import numpy as np
from collections import Counter

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    filename = sys.argv[1]
    f = os.listdir(filename)
    corpus = {}
    for i in range(len(f)):
        txt_file = os.path.join(filename, f[i])
        with open(txt_file, 'r') as txt:
            text = str(txt.read())
            corpus[f[i]] = text
    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    lwr_words = document.lower()
    words = nltk.word_tokenize(lwr_words)
    only_words = []
    stop_words = nltk.corpus.stopwords.words("english")
    for i in range(len(words)):
        word = words[i]
        if word.isalnum() == True and word not in stop_words:
            only_words.append(word)
    return only_words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = []
    all_words = []
    num_docs = 0
    idfs = {}
    for row in documents:
        words.append(documents[row])
        num_docs += 1
    for i in range(len(words)):
        for j in range(len(words[i])):
            word = words[i][j]
            all_words.append(word)
    uq_words = set(all_words)
    unique_words = list(uq_words)
    for i in range(len(unique_words)):
        word1 = unique_words[i]
        idfs[word1] = 0
    for i in range(len(unique_words)):
        for row in documents:
            word1 = unique_words[i]
            if word1 in documents[row]:
                idfs[word1] += 1
    for row in idfs:
        word_docs = idfs[row]
        idf = np.log(num_docs / word_docs)
        idfs[row] = idf

    return idfs
                


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {}
    for word in query:
        for row in files:
            pair = (word, row)
            tf_idfs[pair] = 0
            word_list = files[row]
            d = Counter(word_list)
            count = d[word]
            tf_idfs[pair] = count

    print(tf_idfs)


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()

import math
import nltk
import sys
import os
import string

from nltk.corpus import stopwords

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
    Text_Dict = {}

    with os.scandir(directory) as folder:
        for file in folder:
            textFile = open(file.path, "r", encoding='UTF8')
            Text_Dict[file.name] = textFile.read()
            textFile.close()

    return Text_Dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    token = nltk.word_tokenize(document)
    remember = []

    stop_words = set(stopwords.words('english'))

    for i in range(len(token)):
        token[i] = token[i].lower()

    filtered_sentence = []
    for w in token:
        if w not in string.punctuation:
            filtered_sentence.append(w)

    for v in filtered_sentence:
        if v not in stop_words:
            remember.append(v)

    pre = []
    for v in remember:
        if v not in "==":
            pre.append(v)

    final = []
    for u in pre:
        if u not in "===":
            final.append(u)

    return final


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    for file in documents:
        words = documents[file]
        for word in words:
            f = sum(word in documents[file] for file in documents)
            idf = math.log(len(documents) / f)
            idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.

    tfidfs = dict()
    tf = dict()
    for filename in files:
        tfidfs[filename] = []
        for word in files[filename]:
            if word in tf:
                tf[word] = tf[word] + 1
            else:
                tf[word] = 1
            tfidfs[filename].append((word, tf[word] * idfs[word]))

    for filename in files:
        tfidfs[filename].sort(key=lambda tfidf: tfidf[1], reverse=True)
        tfidfs[filename] = tfidfs[filename][:n]

    print(tfidfs[filename])
    return tfidfs
    """
    tfidf = dict()

    for file in files:
        sum = 0
        for word in query:
            idf = idfs[word]
            sum += files[file].count(word)*idf
        tfidf[file] = sum

    rank = sorted(tfidf.keys(), key = lambda x: tfidf[x], reverse=True)

    rank = list(rank)

    try:
        return rank[0:n]
    except:
        return rank


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = dict()

    for sentence in sentences:
        sum = 0
        words = sentences[sentence]
        count = len(words)
        word_count = 0
        for word in query:
            word_count = words.count(word)
            if word in words:
                sum += idfs[word]

        idf[sentence] = (sum, float(word_count/count))

    rank = sorted(idf.keys(), key=lambda x: idf[x], reverse=True)

    rank = list(rank)
    try:
        return rank[0:n]
    except:
        return rank

if __name__ == "__main__":
    main()

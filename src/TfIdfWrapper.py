from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# This class is a wrapper for the TFidfVectorizer from sklearn, which we use to add functionality
class Tfidf:
    def __init__(self, tokenized_corpus1, tokenized_corpus2, min_occurences=1, ngram_range=(1, 1), max_features=None):
        """
        :param tokenized_corpus1: first corpus to be vectorized. We are going to use this corpus to fit our TF-IDF
        model.
        :param tokenized_corpus2: second corpus to be vectorized. We are going to use this corpus to fit our TF-IDF
        model.
        :param min_occurences: the minimum number of occurences a word must have to be considered as TF-IDF term
        :param ngram_range: a tuple that specifies the ngrams to be considered as TF-IDF terms
        :param max_features: the maximum number of occurences a word must have to be considered as TF-IDF term
        """
        corpus1 = []
        for sentence in tokenized_corpus1:
            corpus1.append(" ".join(sentence))

        corpus2 = []
        for sentence in tokenized_corpus2:
            corpus2.append(" ".join(sentence))

        self._tfidf_vectorizer1 = TfidfVectorizer(min_df=min_occurences,
                                                  ngram_range=ngram_range,
                                                  max_features=max_features)
        self._tfidf_vectorizer1.fit(corpus1)

        self._tfidf_vectorizer2 = TfidfVectorizer(min_df=min_occurences,
                                                  ngram_range=ngram_range,
                                                  max_features=max_features)
        self._tfidf_vectorizer2.fit(corpus2)

    def get_features(self, tokenized_corpus1, tokenized_corpus2, representation='dense'):
        """
        :param tokenized_corpus1:  first tokenized corpus to be vectorized
        :param tokenized_corpus2:  first tokenized corpus to be vectorized
        :param representation: Either 'dense', to create a 2d ndarray, or 'mean' and 'sum' that create 1-d ndarrays
        :return:
        """
        corpus1 = []
        for sentence in tokenized_corpus1:
            corpus1.append(" ".join(sentence))


        corpus2 = []
        for sentence in tokenized_corpus2:
            corpus2.append(" ".join(sentence))

        feature_vectors1 = self._tfidf_vectorizer1.transform(corpus1)
        feature_vectors2 = self._tfidf_vectorizer2.transform(corpus2)

        if representation == 'dense':
            dense_matrix = np.hstack([feature_vectors1.todense(), feature_vectors2.todense()])
            return dense_matrix
        elif representation == 'sum':
            return np.hstack([feature_vectors1.sum(axis=1), feature_vectors2.sum(axis=1)])
        elif representation == 'mean':
            return np.hstack([feature_vectors1.mean(axis=1), feature_vectors2.mean(axis=1)])
        else:
            raise Exception('Wrong Argument')

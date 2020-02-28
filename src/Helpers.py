import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.svm import SVR
from joblib import dump


def rmse(predictions, targets):
    """
    RMSE loss for numpy

    :param predictions: Numpy array of predictions
    :param targets: Numpy array of targets
    :return: RMSE loss
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def svr(X_train, y_train, X_val, y_val, path=None):
    """
    Tries different kernel of SVR and print the accuracy of the validation set

    :param X_train: Training set as a numpy array
    :param y_train: Training targets set as a numpy array
    :param X_val: Validation set as a numpy array
    :param y_val: Validation targets set as a numpy array
    :param path: None: Does not save models / Otherwise, path to directory to save each model
    """
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf_t = SVR(kernel=k)
        clf_t.fit(X_train, y_train)
        print(k)
        if path is not None:
            path2 = path+k+'.joblib'
            dump(clf_t, path2)
        predictions = clf_t.predict(X_val)
        pearson = pearsonr(y_val, predictions)
        print(f'RMSE: {rmse(predictions, y_val)} Pearson {pearson[0]}')
        print()


def write_scores(method_name, scores, filename="../predictions/predictions.txt"):
    """
    Write predictions into a file

    :param method_name: Name of the method used (Not used anymore)
    :param scores: Predictions to write
    :param filename: Path and file name where predictions will be saved
    """
    fn = filename
    print("")
    with open(fn, 'w') as output_file:
        for idx, x in enumerate(scores):
            output_file.write(f"{x}\n")


def get_vector_sentence_bag_of_words(_vocab, sentence, frequency=False):
    """
    Create vector bag of words

    :param _vocab: Vocabulary instance used by the sentence
    :param sentence: Sentence to convert
    :param frequency: True: normalize the vector / False: 1 for appearance, 0 otherwise
    :return: Vector bag of words
    """
    vector = np.zeros(len(_vocab))
    for word in sentence:
        idx = _vocab.word2idx(word)
        if frequency:
            vector[idx] += 1
        else:
            vector[idx] = 1
    if frequency:
        vector = np.divide(vector, len(_vocab))
    return vector


def get_training_bag_of_words_appearance(_vocab1, _vocab2, tokenized_corpus_1, tokenized_corpus_2, frequency=False):
    """
    Create a numpy array for bag of words.

    :param _vocab1: Vocabulary for language 1
    :param _vocab2: Vocabulary for language 2
    :param tokenized_corpus_1: Tokenized corpus of language 1
    :param tokenized_corpus_2: Tokemized corpus of language 2
    :param frequency: If we apply frequency or not (See function get_vector_sentence_bag_of_words)
    :return: Bag of words of both corpus concatenated
    """
    corpus = []
    length = len(tokenized_corpus_1)
    assert(length == len(tokenized_corpus_2))

    for i in range(length):
        vector_1 = get_vector_sentence_bag_of_words(_vocab=_vocab1, sentence=tokenized_corpus_1[i], frequency=frequency)
        vector_2 = get_vector_sentence_bag_of_words(_vocab=_vocab2, sentence=tokenized_corpus_2[i], frequency=frequency)

        # Combine the two tensors
        vector = np.concatenate((vector_1, vector_2), axis=None)

        corpus.append(vector)

    return np.array(corpus)


def tokenize_corpuses(P, corpus_1, corpus_2, punct_1, punct_2, stop_words_1, stop_words_2, test_mode=False, tag=True):
    """
    Tokenize both corpus

    :param P: Preprocessing instance
    :param corpus_1: Corpus of language 1
    :param corpus_2: Corpus of language 2
    :param punct_1: Regex of punctuation to remove for language 1
    :param punct_2: Regex of punctuation to remove for language 2
    :param stop_words_1: List of stop words to remove for language 1
    :param stop_words_2: List of stop words to remove for language 2
    :param test_mode: True if the dataset is a test dataset, False otherwise
    :param tag: True: add tag at the beginning and end of the sentence
    :return: Tuple of tokenized corpus
    """
    tokenized_corpus_1 = P.tokenize(dataset=corpus_1,
                                    punctuation_removal=punct_1,
                                    stop_words=list(stop_words_1), tag_sentence=tag, test_mode=test_mode)
    tokenized_corpus_2 = P.tokenize(dataset=corpus_2,
                                    punctuation_removal=punct_2,
                                    stop_words=list(stop_words_2), tag_sentence=tag, test_mode=test_mode)
    return tokenized_corpus_1, tokenized_corpus_2

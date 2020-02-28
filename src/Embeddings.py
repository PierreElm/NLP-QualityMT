import spacy
from nltk.corpus import stopwords
from scipy import spatial
import torch
import numpy as np


# This serves as a superclass for the classes that are responsible for reading the embeddings.
# The base on this class is the code provided for us from the BaseLine
class SpacyWrapper:
    def __init__(self, lemmatisation=True, stop_words=True, punctuation_removal=False, bert=False, lower=True):
        """
        :param lemmatisation: Boolean, if True lemmatisation is applied
        :param stop_words: Boolean, if True stop_words are removed
        :param punctuation_removal: Boolean, if True punctuation is removed
        :param bert: Boolean, if True Bert embeddings are to be loaded
        :param lower: Boolean, if True the text is tranformed to lowercase
        """

        if not bert:
            self.nlp_en = spacy.load("en300")
            self.nlp_de = spacy.load("de300")
            self.EMB_DIM = 300
        else:
            self.nlp_en = spacy.load("en_trf_xlnetbasecased_lg")
            self.nlp_de = spacy.load("de_trf_bertbasecased_lg")
            self.EMB_DIM = 768

        self.lemmatisation = lemmatisation
        # Pre-processing using spacy
        self.lower = lower
        # Downloading stopwords from the nltk package
        # download('stopwords')   # Stopwords dictionary, run once
        self.punctuation_removal = punctuation_removal
        if stop_words:
            self.stop_words_en = set(stopwords.words('english'))
            self.stop_words_de = set(stopwords.words('german'))
        else:
            self.stop_words_en = set()
            self.stop_words_de = set()


# This Class is responsible for reading embeddings by in the form of one vector for each sentence (the embeddings
# the words of a sentence are averaged)
class PretrainedEmbeddings(SpacyWrapper):
    def get_sentence_emb(self, line, lang):
        """
        :param line: String with the line that is to be mapped to an embedding
        :param lang: 'en' or 'de', the language of the current document
        :return: the vector representation of the line
        """
        if lang == 'en':
            if self.lower:
                text = line.lower()
            else:
                text = line
            if self.lemmatisation:
                l = [token.lemma_ for token in self.nlp_en.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            else:
                l = [token for token in self.nlp_en.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            l = ' '.join([word for word in l if word not in self.stop_words_en])

            return self.nlp_en(l).vector
        elif lang == 'de':
            if self.lower:
                text = line.lower()
            else:
                text = line
            if self.lemmatisation:
                l = [token.lemma_ for token in self.nlp_de.tokenizer(text)]
            else:
                l = [token for token in self.nlp_de.tokenizer(text)]
            l = ' '.join([word for word in l if word not in self.stop_words_de])
            return self.nlp_de(l).vector

    def get_embeddings(self, f, lang, representation='dense'):
        """
        :param f: the file of the corpus
        :param lang: 'en' or 'de', the language of the corpus
        :param representation: 'dense' or 'mean'. If mean the whole corpus is averaged otherwise each sentence is
        a vector
        :return: an ndarray with the embeddings
        """
        file = open(f)
        lines = file.readlines()
        sentences_vectors = []

        for l in lines:
            vec = self.get_sentence_emb(l, lang)
            if vec is not None:
                if representation == 'mean':
                    vec = np.mean(vec)
                elif representation == 'dense':
                    vec = vec
                sentences_vectors.append(vec)
            else:
                print("didn't work :", l)
                sentences_vectors.append(0)
        file.close()
        return np.array(sentences_vectors)


# Class responsible for reading the embeddings and creating a matrix for each sentence by padding or truncating
class NgramEmbedding(PretrainedEmbeddings):
    def __init__(self, lemmatisation=False, stop_words=True, punctuation_removal=True, bert=False, lower=True):
        self.lower = lower
        self.tokenized_corpus_en = []
        self.tokenized_corpus_de = []
        self.max_len = None
        super(NgramEmbedding, self).__init__(lemmatisation, stop_words, punctuation_removal, bert)

    def tokenize_line(self, line, lang):
        """
        :param line: string with the line
        :param lang: 'en' or 'de'
        :return: an list of lists with the tokenized corpus
        """
        if lang == 'en':
            if self.lower:
                text = line.lower()
            else:
                text = line
            if self.lemmatisation:
                l = [token.lemma_ for token in self.nlp_en.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            else:
                l = [token.text for token in self.nlp_en.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            l = [token for token in l if token not in self.stop_words_en]
            self.tokenized_corpus_en.append(l)
        else:
            if self.lower:
                text = line.lower()
            else:
                text = line
            if self.lemmatisation:
                l = [token.lemma_ for token in self.nlp_de.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            else:
                l = [token.text for token in self.nlp_de.tokenizer(text)
                     if (not self.punctuation_removal or not (token.is_punct or token.is_space))]
            l = [token for token in l if token not in self.stop_words_de]
            self.tokenized_corpus_de.append(l)

    def tokenize_corpus(self, f, lang):
        """
        :param f: file name, the location of a corpus to be tokenized
        :param lang: 'en' or 'de'
        :return: the tokenized corpus
        """
        file = open(f)
        lines = file.readlines()

        if lang == 'en':
            self.tokenized_corpus_en = []
        else:
            self.tokenized_corpus_de = []

        for l in lines:
            self.tokenize_line(l, lang)
        file.close()

    def get_max_sentence_size(self):
        return self.max_len

    def _calc_max_len(self):
        max_len = 0
        for line in self.tokenized_corpus_en:
            max_len = max(max_len, len(line))
        for line in self.tokenized_corpus_de:
            max_len = max(max_len, len(line))
        self.max_len = max_len

    def add_unigram(self, matrix, line, lang):
        """
        :param matrix: a list that unigrams will be appended on
        :param line: a tokenized line (a list)
        :param lang: 'en' or 'de'
        :return: the list with the unigram appended
        """
        for token in line:
            if lang == 'en':
                vec = self.nlp_en(token).vector
            else:
                vec = self.nlp_de(token).vector
            matrix.append(vec)
        return matrix

    def add_bigram(self, matrix, line, lang):
        """
        Has not yet implemented. For future functionality this will add the embeddings of the bigrams
        """
        pass

    def add_trigram(self, matrix, line, lang):
        """
        Has not yet implemented. For future functionality this will add the embeddings of the trigrams
        """
        pass

    def get_ngram_embeddings(self, line, lang, ngram, max_len):
        """
        :param line: a tokenized line
        :param lang: 'en' or 'de'
        :param ngram: how many ngrams to take create embeddings for
        :param max_len: the max len as has been determined by the training corpuses
        :return: all the embeddings for a line as a 2d ndarray
        """
        matrix = []
        matrix = self.add_unigram(matrix, line, lang)
        if ngram >= 2:
            max_len += max_len - 1  # add the maxlean of bigrams
            matrix = self.add_bigram(matrix, line, lang)
            if ngram >= 3:
                max_len += max_len - 2  # add the maxlean of bigrams
                matrix = self.add_trigram(matrix, line, lang,)
        matrix = np.array(matrix)
        # Remove extra lines and pad if less lines
        matrix = matrix[:max_len, :]
        return np.pad(matrix, ((0, max_len-matrix.shape[0]), (0, 0)))

    def get_embedding_matrices(self, lang, ngrams=1):
        """
        :param lang: 'en' or 'de'
        :param ngrams: how many ngrams to take create embeddings for
        :return: all the embeddings for a line as a 2d ndarray
        """
        max_len = self.get_max_sentence_size()
        embeddings = []
        if lang == 'en':
            for line in self.tokenized_corpus_en:
                # embs is a numpy array of shape (tokens, 300)
                embs = self.get_ngram_embeddings(line, lang, ngrams, max_len)
                embeddings.append(embs)
        else:
            for line in self.tokenized_corpus_de:
                embs = self.get_ngram_embeddings(line, lang, ngrams, max_len)
                embeddings.append(embs)
        embeddings = np.array(embeddings)
        return embeddings

    def learn_max_len(self, fen, fde):
        """
        :param fen: the file of the english corpus
        :param fde: the file of the german corpus
        :return: the max len of sentence in both files
        """
        self.tokenize_corpus(fen, 'en')
        self.tokenize_corpus(fde, 'de')
        self._calc_max_len()

    def get_embeddings(self, f, lang, representation='mean', ngrams=1):
        """
        :param f: the file of a corpus
        :param lang: 'en' or 'de'
        :param representation: 'mean' or 'dense'
        :param ngrams: 1,2 or 3
        :return: the embeddings matrix of a corpus as a 3d ndarray
        """
        self.tokenize_corpus(f, lang)
        embeddings = self.get_embedding_matrices(lang, ngrams=1)
        return embeddings


# Class that returns an array with the cosine similarities of the embeddings between the english and the german corpuses
# This is not used since after experiements it does not lead to promising results
class CosineSim(SpacyWrapper):
    def get_similarity(self, line1, lang1, line2, lang2):
        """
        :param line1: String
        :param lang1: 'en' or 'de'
        :param line2: String
        :param lang2: 'en' or 'de'
        :return: float, the cosine similarity between the two lines
        """
        if lang1 == 'en':
            text = line1.lower()
            l = [token.lemma_ for token in self.nlp_en.tokenizer(text)]
            l = ' '.join([word for word in l if word not in self.stop_words_en])
            sentence1 = self.nlp_en(l).vector
        else:
            text = line1.lower()
            l = [token.lemma_ for token in self.nlp_de.tokenizer(text)]
            l = ' '.join([word for word in l if word not in self.stop_words_de])
            sentence1 = self.nlp_de(l).vector
        if lang2 == 'en':
            text = line2.lower()
            l = [token.lemma_ for token in self.nlp_en.tokenizer(text)]
            l = ' '.join([word for word in l if word not in self.stop_words_en])
            sentence2 = self.nlp_en(l).vector
        else:
            text = line2.lower()
            l = [token.lemma_ for token in self.nlp_de.tokenizer(text)]
            l = ' '.join([word for word in l if word not in self.stop_words_de])
            sentence2 = self.nlp_de(l).vector
        if np.count_nonzero(sentence1) == 0:
            noise = np.random.normal(-0.01, 0.01, len(sentence1))
            sentence1 += noise
        elif np.count_nonzero(sentence2) == 0:
            noise = np.random.normal(-0.01, 0.01, len(sentence2))
            sentence2 += noise
        return spatial.distance.cosine(sentence1, sentence2)

    def get_similarities(self, f1, lang1, f2, lang2):
        """
        :param f1: the location of one corpus
        :param lang1: 'en' or 'de'
        :param f2: the location of one corpus
        :param lang2: 'en' or 'de'
        :return: a matrix with the cosine similarities of the two corpuses
        """
        file1 = open(f1)
        file2 = open(f2)
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        sims = []

        for i in range(len(lines1)):
            sim = self.get_similarity(lines1[i], lang1, lines2[i], lang2)
            sims.append(sim)

        file1.close()
        file2.close()
        return sims


# This function is responsible for returning the tensors with the embeddings. The main point of this function is that
# the embeddings can be written to a file, from which they can later be read from. This is because mapping the
# sentences to embeddings is extremely time consuming, especially when using the Bert Embeddings
def get_data(embedding_class, params=None, read_embeddings_from_file=False,
             write_to_file=False, dir=None, dtype=torch.float32, stack_as_channels=False):
    """
    :param embedding_class: Class, the class that will be used to create the embeddings
    :param params: Dictionary, the parameters that are going to be passed to the Class constructor
    :param read_embeddings_from_file: Boolean, if True the embeddings are not created again but are read from a file
    :param write_to_file: Boolean, if True the embedding are written to a file
    :param dir: String, The path to file that the embeddings will either be written on or read from
    :param dtype: the datatype of the elemts of the torch tensor that will contain the embeddings
    :param stack_as_channels: Boolean, if True the embeddings of the different languages as stacked as channels in a
    4 dimensional array: (Samples, Channels, Sentences, Embedding Dimensions. Otherwise, they are appended on the
    Embedding dimension
    :return: a tuple of torch tensors for the training,validation and test sets
    """
    if not read_embeddings_from_file:
        # Get the embeddings for the the training, validation and test corpuses located under an en-de/ directory
        # that must be placed on the in the same level as the src/ directory
        if params is None:
            embs = embedding_class()
        else:
            embs = embedding_class(**params)
        # Get training and validation sets
        embs.learn_max_len("../en-de/train.ende.src", "../en-de/train.ende.mt")
        de_train_src = embs.get_embeddings("../en-de/train.ende.src", 'en', representation='dense')
        de_train_mt = embs.get_embeddings("../en-de/train.ende.mt", 'de', representation='dense')
        de_val_src = embs.get_embeddings("../en-de/dev.ende.src", 'en', representation='dense')
        de_val_mt = embs.get_embeddings("../en-de/dev.ende.mt", 'de', representation='dense')

        # Print shapes
        print(f"Training mt: {len(de_train_mt)} Training src: {len(de_train_src)}")
        print()
        print(f"Validation mt: {len(de_val_mt)} Validation src: {len(de_val_src)}")

        de_test_src = embs.get_embeddings("../en-de/test.ende.src", 'en', representation='dense')
        de_test_mt = embs.get_embeddings("../en-de/test.ende.mt", 'de', representation='dense')

        # write the embeddings to a file so they can be read from for efficiency
        if write_to_file:
            np.save(dir + "train.ende.src.npy", de_train_src)
            np.save(dir + "train.ende.mt.npy", de_train_mt)
            np.save(dir + "dev.ende.src.npy", de_val_src)
            np.save(dir + "dev.ende.mt.npy", de_val_mt)
            np.save(dir + "test.ende.src.npy", de_test_src)
            np.save(dir + "test.ende.mt.npy", de_test_mt)

    else:
        de_train_src = np.load(dir + "train.ende.src.npy")
        de_train_mt = np.load(dir + "train.ende.mt.npy")
        de_val_src = np.load(dir + "dev.ende.src.npy")
        de_val_mt = np.load(dir + "dev.ende.mt.npy")

        # Print shapes
        print(f"Training mt: {len(de_train_mt)} Validation src: {len(de_train_src)}")
        print(f"Validation mt: {len(de_val_mt)} Validation src: {len(de_val_src)}")

        de_test_src = np.load(dir + "test.ende.src.npy")
        de_test_mt = np.load(dir + "test.ende.mt.npy")

    X_train = [de_train_src, de_train_mt]
    X_val = [de_val_src, de_val_mt]
    X = [de_test_src, de_test_mt]

    if stack_as_channels:
        # Stacks Embeddings as different channels of the tensor
        X_train_de = torch.tensor(X_train).transpose(0, 1)
        X_val_de = torch.tensor(X_val).transpose(0, 1)
        X_test = torch.tensor(X).transpose(0, 1)
    else:
        # Stacks Embeddings as different features
        X_train_de = torch.tensor(np.concatenate(X_train, axis=1), dtype=dtype)
        X_val_de = torch.tensor(np.concatenate(X_val, axis=1), dtype=dtype)
        X_test = torch.tensor(np.concatenate(X, axis=1), dtype=dtype)

    f_train_scores = open("../en-de/train.ende.scores", 'r')
    de_train_scores = f_train_scores.readlines()
    f_train_scores.close()

    f_val_scores = open("../en-de/dev.ende.scores", 'r')
    de_val_scores = f_val_scores.readlines()
    f_train_scores.close()

    train_scores = np.array(de_train_scores).astype(float)
    y_train_de = torch.tensor(train_scores, dtype=dtype)

    val_scores = torch.tensor(np.array(de_val_scores).astype(float), dtype=dtype)
    y_val_de = val_scores

    return X_train_de, y_train_de, X_val_de, y_val_de, X_test

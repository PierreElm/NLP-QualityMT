import re
import torch


class PreProcessing:
    def __init__(self):
        self.max_len = 0

    @staticmethod
    def load_dataset(file):
        """
        Load dataset from a file

        :param file: Path to the file
        :return: Return the corpus, 1 sentence per line
        """
        file = open(file, 'r', encoding="utf8")
        sentences = file.readlines()
        # Return all corpus by sentences
        return sentences

    @staticmethod
    def split_with_punctuation(punctuation, tokenized_sentence, token, stop_words):
        """
        Split punctuation in a token and append it to the tokenized sentence

        :param punctuation: List of punctuation to take into account
        :param tokenized_sentence: Tokenized sentence where we will append the token
        :param token: Token to check for split
        :param stop_words: Stop words that we should not include into the tokenized sentence
        :return: Tokenized sentence with the token and punctuation appended
        """
        for punct in punctuation:
            if punct in token:
                words = token.split(punct)

                # We add the first word
                if words[0] != '' and (stop_words is None or words[0] not in stop_words):
                    tokenized_sentence.append(words[0])

                # We add the punctuation
                if stop_words is None or punct not in stop_words:
                    tokenized_sentence.append(punct)

                # If there was a word after the punctuation, we add it aswell
                if words[1] != '' and (stop_words is None or words[1] not in stop_words):
                    tokenized_sentence.append(words[1])

                # We return the tokenized sentence updated.
                return tokenized_sentence

        # If no punctuation into token.
        tokenized_sentence.append(token)
        return tokenized_sentence

    def tokenize(self, dataset, punctuation_removal='[\s]', punctuation_split=None, stop_words=None, test_mode=False,
                 tag_sentence=False):
        """
        Tokenize a dataset

        :param dataset: Dataset to tokenize
        :param punctuation_removal: Punctuation to remove (Regex)
        :param punctuation_split: List of punctuation to split
        :param stop_words: List of stop words to remove
        :param test_mode: True if the dataset is a test set. Do not take the max length of it into account
        :param tag_sentence: True if we want to add tag at the beginning and the end of each sentence, False otherwise
        :return: The tokenized corpus
        """
        # Tokenized dataset
        tokenized_corpus = []

        for sentence in dataset:
            # Tokenized sentence
            tokenized_sentence = []
            if tag_sentence:
                tokenized_sentence.append('<s>')

            # Remove \n on sentence
            sentence = sentence.strip()

            sentence = re.split(punctuation_removal, sentence)

            for token in sentence:
                # If we are pre-processing test dataset and the sentence is longer than the sentences we had in training
                if test_mode and len(tokenized_sentence) >= self.max_len:
                    break

                token = token.lower()

                # If the token is empty we continue
                if token == '':
                    continue

                # If the token is a stop word, we do not add it to the tokenized sentence
                if stop_words is not None and token in stop_words:
                    continue

                # If we need to split some punctuation
                if punctuation_split is not None:
                    # We look for punctuation in the token.
                    tokenized_sentence = PreProcessing.split_with_punctuation(punctuation_split,
                                                                              tokenized_sentence,
                                                                              token,
                                                                              stop_words)
                else:
                    tokenized_sentence.append(token)

            if tag_sentence:
                tokenized_sentence.append('</s>')

            # We add the tokenized sentence to the tokenized corpus
            if len(tokenized_sentence) > self.max_len and test_mode is False:
                self.max_len = len(tokenized_sentence)

            tokenized_corpus.append(tokenized_sentence)

        return tokenized_corpus

    @staticmethod
    def tokenize_bigram(tokenized_corpus):
        """
        Tokenization for bi-gram
        :param tokenized_corpus: Tokernized corpus that will be used to generate the bi-gram
        :return: Bi-gram corpus tokenized
        """
        corpus_bigram = []
        for sentence in tokenized_corpus:
            sentence_bigram = []
            for i in range(0, len(sentence) - 1):
                bigram = (sentence[i], sentence[i+1])
                sentence_bigram.append(bigram)
            corpus_bigram.append(sentence_bigram)
        return corpus_bigram

    @staticmethod
    def tokenize_trigram(tokenized_corpus):
        """
        Tokenization for tri-gram
        :param tokenized_corpus: Tokernized corpus that will be used to generate the tri-gram
        :return: Tri-gram corpus tokenized
        """
        corpus_trigram = []
        for sentence in tokenized_corpus:
            sentence_trigram = []
            for i in range(0, len(sentence) - 2):
                trigram = (sentence[i], sentence[i + 1], sentence[i + 2])
                sentence_trigram.append(trigram)
            corpus_trigram.append(sentence_trigram)
        return corpus_trigram

    @staticmethod
    def get_special_characters(dataset):
        """
        Return all the special characters of the dataset
        :param dataset: Dataset to analyse
        :return: Sorted list of the characters
        """
        symbols = set()
        for sentence in dataset:
            punct = (" ".join(re.split("[\s/^\w+]+", sentence, re.UNICODE))).split()
            fixed_punct = []
            for symbol in punct:
                if symbol == "..." or len(symbol) == 1:
                    # the only sequence of symbols that is recognised as a symbol is the three dots
                    fixed_punct.append(symbol)
                else:
                    fixed_punct + list(symbol)
            symbols.update(fixed_punct)
        return sorted(list(symbols), key=len, reverse=True)

    @staticmethod
    def get_regex_special_characters(dataset):
        """
        Produce the regex with the characters that need to be removed
        :param dataset: Dataset to analize
        :return: Regex
        """
        special_chars = PreProcessing.get_special_characters(dataset)
        regex = "[\s]"
        for char in special_chars:
            regex += "|[" + char + "]"
        return regex

    def sentence_to_tensor(self, _vocab, sentence):
        """
        Convert a sente to a tensor with index of each word
        :param _vocab: Vocabulary to use to get indices
        :param sentence: Sentence to convert
        :return: Tensor with indices of word w.r.t the Vocabulary
        """
        idxs = []
        idxs.extend(_vocab.convert_words_to_idxs(sentence))
        tensor = torch.LongTensor(idxs)
        difference_padding = self.max_len - len(tensor)
        tensor = torch.nn.functional.pad(tensor, [0, difference_padding])

        return tensor

    def get_tensor_set_for_regression(self, _vocab_1, _vocab_2, tokenized_corpus_1, tokenized_corpus_2):
        """
        Tensor of both corpus concatenated
        :param _vocab_1: Vocabulary of language 1
        :param _vocab_2: Vocabulary of language 2
        :param tokenized_corpus_1: Tokenized corpus 1
        :param tokenized_corpus_2: Tokenized corpus 2
        :return: Concatenation fo both corpus as tensor
        """

        tensor_corpus = None
        assert(len(tokenized_corpus_1) == len(tokenized_corpus_2))

        for i in range(len(tokenized_corpus_1)):
            sentence_1 = self.sentence_to_tensor(_vocab_1, tokenized_corpus_1[i])
            sentence_2 = self.sentence_to_tensor(_vocab_2, tokenized_corpus_2[i])

            # Combine two tensors
            sentence_tensor = torch.cat((sentence_1, sentence_2))

            # Append it to tensor corpus
            if tensor_corpus is None:
                tensor_corpus = sentence_tensor.view(1, sentence_tensor.shape[0])
            else:
                tensor_corpus = torch.cat((tensor_corpus, sentence_tensor.view(1, sentence_tensor.shape[0])))

        return tensor_corpus

    @staticmethod
    def get_batches(data_tensor, batch_size=64):
        """
        Produces batches
        :param data_tensor: Tensor to divide
        :param batch_size: Size of each batch
        :return: Batches of the dataset
        """
        # Get the number of training
        n_samples = data_tensor.shape[0]

        # Hack: discard the last uneven batch for simplicity
        n_batches = n_samples // batch_size
        n_samples = n_batches * batch_size
        # Split nicely into batches, i.e. (n_batches, batch_size, context_size + 1)
        # The final element in each row is the ID of the true label to predict
        x_y = data_tensor[:n_samples].view(n_batches, batch_size, -1)

        return x_y

    @staticmethod
    def dictionary_frequencies(tokenized_corpus):
        """
        Build a frequency dictionary of words in the corpus
        :param tokenized_corpus: Tokenized corpus
        :return: Frequency dictionary
        """
        dictionary = {}
        for sentence in tokenized_corpus:
            for token in sentence:
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1

        return dictionary

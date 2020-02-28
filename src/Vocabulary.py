class Vocabulary(object):
    """Data structure representing the vocabulary of a corpus."""

    def __init__(self, count_dictionary, min_count=1):
        # Mapping from tokens to integers
        self._word2idx = {}

        # Reverse-mapping from integers to tokens
        self.idx2word = []
        # Frequencies dictionary for Happax legomenon
        self.count_dictionary = count_dictionary
        self.minimum_count = min_count

        # 0-padding token
        self.add_word('<pad>')
        # sentence start
        self.add_word('<s>')
        # sentence end
        self.add_word('</s>')
        # Unknown words
        self.add_word('<unk>')

        self._unk_idx = self._word2idx['<unk>']

    def word2idx(self, word):
        """Returns the integer ID of the word or <unk> if not found."""
        return self._word2idx.get(word, self._unk_idx)

    def add_word(self, word):
        """Adds the `word` into the vocabulary."""
        if word not in self._word2idx:
            self.idx2word.append(word)
            self._word2idx[word] = len(self.idx2word) - 1

    def build_from_file(self, fname):
        """Builds a vocabulary from a given corpus file."""
        with open(fname) as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    self.add_word(word)

    def build_from_token(self, corpus, min_count=None, dictionary=None):
        min_count = self.minimum_count if min_count is None else min_count
        dictionary = self.count_dictionary if dictionary is None else dictionary
        for sentence in corpus:
            for word in sentence:
                if dictionary[word] > min_count:
                    self.add_word(word)

    def build_from_token_bi_gram(self, corpus):
        for sentence in corpus:
            for i in range(len(sentence) - 1):
                word_1 = sentence[i] if (self.count_dictionary[sentence[i]] > self.minimum_count) else '<unk>'
                word_2 = sentence[i+1] if (self.count_dictionary[sentence[i+1]] > self.minimum_count) else '<unk>'
                if word_2 != '<unk>' and word_1 != '<unk>':
                    self.add_word((word_1, word_2))

    def convert_idxs_to_words(self, idxs):
        """Converts a list of indices to words."""
        return ' '.join(self.idx2word[idx] for idx in idxs)

    def convert_words_to_idxs(self, words):
        """Converts a list of words to a list of indices."""
        return [self.word2idx(w) for w in words]

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.idx2word)

    def __repr__(self):
        return "Vocabulary with {} items".format(self.__len__())

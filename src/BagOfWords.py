from nltk import download
from PreProcessing import PreProcessing
from Vocabulary import Vocabulary
from Helpers import *
from NN import MLP
import torch
from nltk.corpus import stopwords


class BagOfWords:

    def __init__(self, P, corpus_1, corpus_2, y_train, stop_words_1, stop_words_2, corpus_val_1, corpus_val_2, y_val,
                 n_gram='unigram', frequency=False, min_count=2, min_count_gram=1):
        """
        BagOfWords Constructor

        :param P: Preprocessing instance used on the training sets
        :param corpus_1: Training corpus for language 1
        :param corpus_2: Training corpus for language 2
        :param y_train: Training labels
        :param stop_words_1: Stop words for language 1
        :param stop_words_2: Stop words for language 2
        :param corpus_val_1: Validation corpus for language 1
        :param corpus_val_2: Validation corpus for language 2
        :param y_val: Validation labels
        :param n_gram: Option for the construction of the vocabulary (See READLE.md for details)
        :param frequency: True: normalize the bag of words with frequency of words in the sentence.
                          False: 1 if word in the sentence, 0 otherwise.
        :param min_count: Do not add words in vocabulary that are not present more than min_count.
        :param min_count_gram: Do not add n-gram in vocabulary that are not present more than min_count_gram.
        """

        # General parameters
        self.y_train = y_train
        self.y_val = y_val
        self.model = None
        self.device = None
        self.optimizer = None
        self.frequency = frequency
        self.n_gram = n_gram
        self.P = P
        self.pred = None
        self.X_test = None
        self.predictions = None

        # Get punctuation
        self.punct_1 = P.get_regex_special_characters(corpus_1)
        self.punct_2 = P.get_regex_special_characters(corpus_2)
        # Stop words
        self.stop_words_1 = stop_words_1
        self.stop_words_2 = stop_words_2

        # Tokenize corpus
        tokenized_corpus_1, tokenized_corpus_2 = tokenize_corpuses(P=P, corpus_1=corpus_1, corpus_2=corpus_2,
                                                                   punct_1=self.punct_1, punct_2=self.punct_2,
                                                                   stop_words_1=stop_words_1, stop_words_2=stop_words_2)

        # Build vocabularies
        count_dict_1 = P.dictionary_frequencies(tokenized_corpus_1)
        vocab_1 = Vocabulary(count_dictionary=count_dict_1, min_count=min_count)
        count_dict_2 = P.dictionary_frequencies(tokenized_corpus_2)
        vocab_2 = Vocabulary(count_dictionary=count_dict_2, min_count=min_count)
        vocab_1.build_from_token(tokenized_corpus_1)
        vocab_2.build_from_token(tokenized_corpus_2)

        # If bigram mode we build bi-gram vocabulary as well
        if n_gram == 'bigram':
            t1 = P.tokenize_bigram(tokenized_corpus_1)
            count_dict_1 = P.dictionary_frequencies(t1)
            vocab_1.build_from_token(corpus=t1, min_count=min_count_gram, dictionary=count_dict_1)
            t2 = P.tokenize_bigram(tokenized_corpus_2)
            count_dict_2 = P.dictionary_frequencies(t2)
            vocab_2.build_from_token(corpus=t2, min_count=min_count_gram, dictionary=count_dict_2)
            for i in range(len(tokenized_corpus_1)):
                tokenized_corpus_1[i] += t1[i]
                tokenized_corpus_2[i] += t2[i]

        # If trigram mode, we build bi-gram and tri-gram vocabulary as well
        if n_gram == 'trigram':
            # Bigram
            t1 = P.tokenize_bigram(tokenized_corpus_1)
            count_dict_1 = P.dictionary_frequencies(t1)
            vocab_1.build_from_token(corpus=t1, min_count=min_count_gram, dictionary=count_dict_1)
            t2 = P.tokenize_bigram(tokenized_corpus_2)
            count_dict_2 = P.dictionary_frequencies(t2)
            vocab_2.build_from_token(corpus=t2, min_count=min_count_gram, dictionary=count_dict_2)
            # Trigram
            t3 = P.tokenize_trigram(tokenized_corpus_1)
            count_dict_1 = P.dictionary_frequencies(t3)
            vocab_1.build_from_token(corpus=t3, min_count=min_count_gram, dictionary=count_dict_1)
            t4 = P.tokenize_trigram(tokenized_corpus_2)
            count_dict_2 = P.dictionary_frequencies(t4)
            vocab_2.build_from_token(corpus=t4, min_count=min_count_gram, dictionary=count_dict_2)
            for i in range(len(tokenized_corpus_1)):
                tokenized_corpus_1[i] += t1[i] + t3[i]
                tokenized_corpus_2[i] += t2[i] + t4[i]

        # Build X_train
        self.X_train = get_training_bag_of_words_appearance(_vocab1=vocab_1, _vocab2=vocab_2,
                                                            tokenized_corpus_1=tokenized_corpus_1,
                                                            tokenized_corpus_2=tokenized_corpus_2,
                                                            frequency=frequency)
        # Tokenize validation sets
        tokenized_val_corpus_1, tokenized_val_corpus_2 = tokenize_corpuses(P=P,
                                                                           corpus_1=corpus_val_1, corpus_2=corpus_val_2,
                                                                           punct_1=self.punct_1, punct_2=self.punct_2,
                                                                           stop_words_1=stop_words_1,
                                                                           stop_words_2=stop_words_2)
        # If bigram mode, tokenize for bi-gram as well
        if n_gram == 'bigram':
            t1 = P.tokenize_bigram(tokenized_val_corpus_1)
            t2 = P.tokenize_bigram(tokenized_val_corpus_2)
            for i in range(len(tokenized_val_corpus_1)):
                tokenized_val_corpus_1[i] += t1[i]
                tokenized_val_corpus_2[i] += t2[i]

        # If trigram mode, tokenize for bi-gram and tri-gram as well
        if n_gram == 'trigram':
            t1 = P.tokenize_bigram(tokenized_val_corpus_1)
            t2 = P.tokenize_bigram(tokenized_val_corpus_2)
            t3 = P.tokenize_trigram(tokenized_val_corpus_1)
            t4 = P.tokenize_trigram(tokenized_val_corpus_2)
            for i in range(len(tokenized_val_corpus_1)):
                tokenized_val_corpus_1[i] += t1[i] + t3[i]
                tokenized_val_corpus_2[i] += t2[i] + t4[i]

        # Build X_val
        self.X_val = get_training_bag_of_words_appearance(_vocab1=vocab_1, _vocab2=vocab_2,
                                                          tokenized_corpus_1=tokenized_val_corpus_1,
                                                          tokenized_corpus_2=tokenized_val_corpus_2,
                                                          frequency=frequency)
        # Vocabularies
        self.vocab_1 = vocab_1
        self.vocab_2 = vocab_2

        # Information after creation of the instance
        print(n_gram + " model:")
        print("Shape training set: ", self.X_train.shape)
        print("Shape validation set: ", self.X_val.shape)
        print("Shape training scores: ", y_train.shape)
        print("Shape validation scores: ", y_val.shape)
        print()

    # Call SVR method that tries different parameters
    def try_svr(self):
        """
        Perform SVR with multiple parameters on training set and use validation set to evaluate the performances.
        The function will print the different scores.
        """
        svr(X_train=self.X_train, y_train=self.y_train, X_val=self.X_val, y_val=self.y_val)

    def train_model_svr(self, kernel='rbf'):
        """
        Train the model using SVR with the selected kernel

        :param kernel: Kernel used to train SVR (‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’)
        :return: Print performances on validation set
        """
        self.model = SVR(kernel=kernel)
        self.model.fit(self.X_train, self.y_train)

        # Validation
        self.pred = self.model.predict(self.X_val)
        pearson = pearsonr(y_val, self.pred)
        print(f'RMSE: {rmse(self.pred, self.y_val)} Pearson {pearson[0]}')
        print()

    def build_test_set(self, corpus_test_1, corpus_test_2):
        """
        Tokenize the test set

        :param corpus_test_1: Test set of language 1
        :param corpus_test_2: Test set of language 2
        :return: Print the shape of the test set
        """
        tokenized_test_corpus_1, tokenized_test_corpus_2 = tokenize_corpuses(P=P,
                                                                             corpus_1=corpus_test_1,
                                                                             corpus_2=corpus_test_2,
                                                                             punct_1=self.punct_1,
                                                                             punct_2=self.punct_2,
                                                                             stop_words_1=self.stop_words_1,
                                                                             stop_words_2=self.stop_words_2)
        if self.n_gram == 'bigram':
            t1 = P.tokenize_bigram(tokenized_test_corpus_1)
            t2 = P.tokenize_bigram(tokenized_test_corpus_2)
            for i in range(len(tokenized_test_corpus_1)):
                tokenized_test_corpus_1[i] += t1[i]
                tokenized_test_corpus_2[i] += t2[i]

        if self.n_gram == 'trigram':
            t1 = P.tokenize_trigram(tokenized_test_corpus_1)
            t2 = P.tokenize_trigram(tokenized_test_corpus_2)
            t3 = P.tokenize_bigram(tokenized_test_corpus_1)
            t4 = P.tokenize_bigram(tokenized_test_corpus_2)
            for i in range(len(tokenized_test_corpus_1)):
                tokenized_test_corpus_1[i] += t1[i] + t3[i]
                tokenized_test_corpus_2[i] += t2[i] + t4[i]

        self.X_test = get_training_bag_of_words_appearance(_vocab1=self.vocab_1,
                                                           _vocab2=self.vocab_2,
                                                           tokenized_corpus_1=tokenized_test_corpus_1,
                                                           tokenized_corpus_2=tokenized_test_corpus_2,
                                                           frequency=self.frequency)
        print("Shape test set: ", self.X_test.shape)

    def make_prediction_svr(self):
        """
        Process the test set and make predictions

        :return: Write a file with the predictions
        """
        if self.X_test is None:
            raise Exception('Test set undefined.')
        self.predictions = self.model.predict(self.X_test)
        if self.frequency:
            write_scores("SVR", self.predictions, filename='../predictions/bow_frequency_svr.txt')
        else:
            write_scores("SVR", self.predictions, filename='../predictions/bow_svr.txt')

    def create_model(self, gpu=True):
        """
        Create the Multi layer Perceptron model

        :param gpu: If we use GPU or not
        """
        # Create tensor
        self.X_train = torch.tensor(self.X_train)
        self.y_train = torch.from_numpy(self.y_train.reshape(self.y_train.shape[0], 1)).float()

        # Create batches
        self.X_train = self.P.get_batches(self.X_train, batch_size=512)
        self.y_train = self.P.get_batches(self.y_train, batch_size=512)

        # Set device
        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        input_dimension = self.X_train.shape[2]
        self.model = MLP(input_dimension=input_dimension)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_model(self, epoch=5):
        """
        Train the model

        :param epoch: Number of epoch to train
        """
        self.model = self.model.to(device=self.device)

        for e in range(epoch):
            train_loss = 0.0
            pearson_train = 0.0
            # Shuffle batches
            permutation = torch.randperm(self.X_train.size()[0])
            for t in range(len(self.X_train)):
                self.model.train()  # Put model on training mode

                x = self.X_train[permutation[t]].to(device=self.device, dtype=torch.float32)
                y = self.y_train[permutation[t]].to(device=self.device, dtype=torch.float)

                # Get prediction
                scores = self.model(x)
                # Compute loss
                pearson = pearsonr(np.squeeze(y.data.numpy()), np.squeeze(scores.data.numpy()))
                loss = torch.sqrt(((scores - y) ** 2).mean())  # RMSE
                train_loss += loss
                pearson_train += pearson[0]
                # loss = F.mse_loss(scores, y) # MSE loss
                # Zero out all of the gradients for the variables which the optimizer will update.
                self.optimizer.zero_grad()
                # Back propagation
                loss.backward()
                # Update the parameters of the model using the gradients
                self.optimizer.step()

                # Print the loss
                # if t % 100 == 0:
                #    print('Epoch: %d, Iteration %d, loss = %.4f, pearson %.4f' % (e, t, loss.item(), pearson[0]))
                #    print()
            self.check_accuracy(e, train_loss/len(self.X_train), pearson_train/len(self.X_train))

    def check_accuracy(self, epoch=0, train_loss=0.0, pearson_train=0.0):
        """
        Check accuracy of the model on validation set

        :param epoch: Current epoch
        :param train_loss: Current training loss
        :param pearson_train: Current training pearson value
        :return: Print the accuracy of the model
        """
        total_diff = 0
        num_samples = 0
        predictions = np.array(())
        targets = np.array(())

        # Create tensor
        X_val = torch.tensor(self.X_val)
        y_val = torch.from_numpy(self.y_val.reshape(self.y_val.shape[0], 1)).float()

        self.model.eval()
        with torch.no_grad():
            for t in range(X_val.shape[0]):
                x = X_val[t].to(device=self.device, dtype=torch.float32)
                y = y_val[t].to(device=self.device, dtype=torch.float)
                scores = self.model(x)
                total_diff += abs((scores - y)).sum()
                num_samples += scores.size(0)
                predictions = np.append(predictions, np.squeeze(scores.data.numpy()))
                targets = np.append(targets, np.squeeze(y.data.numpy()))

            pearson_val = pearsonr(targets, predictions)
            val_loss = rmse(predictions, targets)
            print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f}| Pearson Train: {pearson_train:.6f} | '
                  f'Pearson Val: {pearson_val[0]:.6f} | Val Loss: {val_loss:.3f}|')

    def predict(self):
        """
        Build predictions on test set
        """
        predictions = np.array(())
        # Create tensor
        X_test = torch.tensor(self.X_test)

        self.model.eval()
        with torch.no_grad():
            for t in range(X_test.shape[0]):
                x = X_test[t].to(device=self.device, dtype=torch.float32)
                scores = self.model(x)
                predictions = np.append(predictions, np.squeeze(scores.data.numpy()))

        np.savetxt('../predictions/BoW.txt', predictions, fmt='%.12f')


# Downloading stopwords from the nltk package
download('stopwords')   # Stopwords dictionary, run once
stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))

# Load training sets
P = PreProcessing()
corpus_en = P.load_dataset('../en-de/train.ende.src')
corpus_de = P.load_dataset('../en-de/train.ende.mt')
# Load validation sets
corpus_val_en = P.load_dataset('../en-de/dev.ende.src')
corpus_val_de = P.load_dataset('../en-de/dev.ende.mt')
# Load test corpus
corpus_test_en = P.load_dataset('../en-de/test.ende.src')
corpus_test_de = P.load_dataset('../en-de/test.ende.mt')
# Scores
y_train = np.loadtxt('../en-de/train.ende.scores')
y_val = np.loadtxt('../en-de/dev.ende.scores')


"""
######### Create uni-gram model #########
unigram_model = BagOfWords(P=P, corpus_1=corpus_en, corpus_2=corpus_de, y_train=y_train, stop_words_1=stop_words_en,
                           stop_words_2=stop_words_de, corpus_val_1=corpus_val_en, corpus_val_2=corpus_val_de,
                           y_val=y_val, n_gram='unigram')


######### Create bi-gram model #########                        
bigram_model = BagOfWords(P=P, corpus_1=corpus_en, corpus_2=corpus_de, y_train=y_train, stop_words_1=stop_words_en,
                          stop_words_2=stop_words_de, corpus_val_1=corpus_val_en, corpus_val_2=corpus_val_de,
                          y_val=y_val, n_gram='bigram', min_count=2)


######### Create tri-gram model #########                        
model = BagOfWords(P=P, corpus_1=corpus_en, corpus_2=corpus_de, y_train=y_train, stop_words_1=stop_words_en,
                   stop_words_2=stop_words_de, corpus_val_1=corpus_val_en, corpus_val_2=corpus_val_de,
                   y_val=y_val, n_gram='trigram', min_count=2)


######### Test kernels with SVR #########
model.try_svr()

######### Train with svr #########
model.train_model_svr('rbf')
"""

# Full example using MLP
model = BagOfWords(P=P, corpus_1=corpus_en, corpus_2=corpus_de, y_train=y_train, stop_words_1=stop_words_en,
                   stop_words_2=stop_words_de, corpus_val_1=corpus_val_en, corpus_val_2=corpus_val_de,
                   y_val=y_val, n_gram='unigram', min_count=1)
model.create_model()
model.train_model(epoch=20)
model.build_test_set(corpus_test_en, corpus_test_de)
model.predict()

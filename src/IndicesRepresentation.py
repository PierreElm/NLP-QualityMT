from nltk import download
from Vocabulary import Vocabulary
from PreProcessing import PreProcessing
from Helpers import *
from sklearn.linear_model import LinearRegression

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

# Get punctuation
punct_en = P.get_regex_special_characters(corpus_en)
punct_de = P.get_regex_special_characters(corpus_de)

# Tokenized corpus
tokenized_corpus_en, tokenized_corpus_de = tokenize_corpuses(P=P, corpus_1=corpus_en, corpus_2=corpus_de,
                                                             punct_1=punct_en, punct_2=punct_de,
                                                             stop_words_1=stop_words_en, stop_words_2=stop_words_de)

tokenized_corpus_val_en, tokenized_corpus_val_de = tokenize_corpuses(P=P,
                                                                     corpus_1=corpus_val_en, corpus_2=corpus_val_de,
                                                                     punct_1=punct_en, punct_2=punct_de,
                                                                     stop_words_1=stop_words_en,
                                                                     stop_words_2=stop_words_de, test_mode=True)

tokenized_corpus_test_en, tokenized_corpus_test_de = tokenize_corpuses(P=P,
                                                                       corpus_1=corpus_test_en, corpus_2=corpus_test_de,
                                                                       punct_1=punct_en, punct_2=punct_de,
                                                                       stop_words_1=stop_words_en,
                                                                       stop_words_2=stop_words_de, test_mode=True)

# Build vocabularies
count_dict_en = P.dictionary_frequencies(tokenized_corpus_en)
vocab_en = Vocabulary(count_dictionary=count_dict_en, min_count=0)
count_dict_de = P.dictionary_frequencies(tokenized_corpus_de)
vocab_de = Vocabulary(count_dictionary=count_dict_de, min_count=0)
vocab_en.build_from_token(tokenized_corpus_en)
vocab_de.build_from_token(tokenized_corpus_de)

# Build datasets
X_train = P.get_tensor_set_for_regression(vocab_en, vocab_de, tokenized_corpus_en, tokenized_corpus_de).numpy()
X_val = P.get_tensor_set_for_regression(vocab_en, vocab_de, tokenized_corpus_val_en, tokenized_corpus_val_de).numpy()
X_test = P.get_tensor_set_for_regression(vocab_en, vocab_de, tokenized_corpus_test_en, tokenized_corpus_test_de).numpy()

print("Training shape: ", X_train.shape)
print("Validation shape: ", X_val.shape)

# Apply linear regression
lr = LinearRegression().fit(X_train, y_train)

# Validation
predictions = lr.predict(X_val)
print("RMSE on validation set: ", rmse(predictions, y_val))
print("Pearson on validation set: ", pearsonr(predictions, y_val)[0])

# Predictions on test set
# Re-train the linear regression with training + validation set
X_train = np.concatenate((X_train, X_val))
y_train = np.concatenate((y_train, y_val))
print("Final training shape: ", X_train.shape)
print("Final scores shape: ", y_train.shape)
lr = LinearRegression().fit(X_train, y_train)

predictions = lr.predict(X_test)
write_scores("LinearRegression", scores=predictions, filename="../predictions/IndicesRepresentations.txt")

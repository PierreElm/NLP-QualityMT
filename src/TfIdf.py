from nltk import download
from PreProcessing import PreProcessing
from Helpers import *
from TfIdfWrapper import Tfidf

# selecting if we remove stop words or not
from nltk.corpus import stopwords

remove_stop_words = False

if remove_stop_words:
    # Downloading stopwords from the nltk package
    download('stopwords')   # Stopwords dictionary, run once

    stop_words_en = set(stopwords.words('english'))
    stop_words_de = set(stopwords.words('german'))

else:
    stop_words_en = set()
    stop_words_de = set()

# Load training sets
P = PreProcessing()
corpus_en = P.load_dataset('../en-de/train.ende.src')
punct_en = P.get_special_characters(corpus_en)
corpus_de = P.load_dataset('../en-de/train.ende.mt')
punct_de = P.get_special_characters(corpus_de)

# tokenize and preprocess the corpuses
tokenized_corpus_en = P.tokenize(dataset=corpus_en,
                                 punctuation_removal='[\s"\']',
                                 punctuation_split=punct_en,
                                 stop_words=stop_words_en)
tokenized_corpus_de = P.tokenize(dataset=corpus_de,
                                 punctuation_removal='[\s"\']',
                                 punctuation_split=punct_de,
                                 stop_words=stop_words_de)

# select parameters for the TF-IDF model such as the ngram type
ngram_range = (1, 3)
representation = 'mean'
min_occurences = 1
max_features = None

# Build English and german TFidf models
tfidf = Tfidf(tokenized_corpus1=tokenized_corpus_en,
              tokenized_corpus2=tokenized_corpus_de,
              min_occurences=min_occurences,
              ngram_range=ngram_range,
              max_features=max_features)

# Build X_train
X_train = tfidf.get_features(tokenized_corpus1=tokenized_corpus_en,
                             tokenized_corpus2=tokenized_corpus_de,
                             representation=representation)


# Build X_val
corpus_val_en = P.load_dataset('../en-de/dev.ende.src')
corpus_val_de = P.load_dataset('../en-de/dev.ende.mt')
tokenized_val_corpus_en = P.tokenize(dataset=corpus_val_en,
                                     punctuation_removal='[\s"\']',
                                     punctuation_split=punct_en,
                                     stop_words=punct_en + list(stop_words_en),
                                     test_mode=True)
tokenized_val_corpus_de = P.tokenize(dataset=corpus_val_de,
                                     punctuation_removal='[\s"\']',
                                     punctuation_split=punct_de,
                                     stop_words=punct_de + list(stop_words_de),
                                     test_mode=True)

X_val = tfidf.get_features(tokenized_corpus1=tokenized_val_corpus_en,
                           tokenized_corpus2=tokenized_val_corpus_de,
                           representation=representation)

# Scores
y_train = np.loadtxt('../en-de/train.ende.scores')
y_val = np.loadtxt('../en-de/dev.ende.scores')

print("Shape training set: ", X_train.shape)
print("Shape validation set: ", X_val.shape)
print("Shape training scores: ", y_train.shape)
print("Shape validation scores: ", y_val.shape)

# Call SVR
svr(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

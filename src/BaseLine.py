import spacy
from nltk import download
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from Helpers import *


# Be sure to run :
# python -m spacy download en_core_web_md
# python -m spacy link en_core_web_md en300
#
# python -m spacy download de_core_news_md
# python -m spacy link de_core_news_md de300


# Import spacy en and de
nlp_en = spacy.load("en300")
nlp_de = spacy.load("de300")

# Pre-processing using spacy

# Downloading stopwords from the nltk package
download('stopwords')   # Stopwords dictionary, run once

stop_words_en = set(stopwords.words('english'))
stop_words_de = set(stopwords.words('german'))


def get_sentence_emb(line, nlp, lang):
    if lang == 'en':
        text = line.lower()
        l = [token.lemma_ for token in nlp.tokenizer(text)]
        l = ' '.join([word for word in l if word not in stop_words_en])

    elif lang == 'de':
        text = line.lower()
        l = [token.lemma_ for token in nlp.tokenizer(text)]
        l = ' '.join([word for word in l if word not in stop_words_de])

    sen = nlp(l)
    return nlp(l).vector


def get_embeddings(f, nlp, lang):
    file = open(f)
    lines = file.readlines()
    sentences_vectors = []

    for l in lines:
        vec = get_sentence_emb(l, nlp, lang)
        if vec is not None:
            vec = np.mean(vec)
            sentences_vectors.append(vec)
        else:
            print("didn't work :", l)
            sentences_vectors.append(0)
    return sentences_vectors


# Get training and validation sets
de_train_src = get_embeddings("../en-de/train.ende.src", nlp_en, 'en')
de_train_mt = get_embeddings("../en-de//train.ende.mt", nlp_de, 'de')

f_train_scores = open("../en-de/train.ende.scores", 'r')
de_train_scores = f_train_scores.readlines()

de_val_src = get_embeddings("../en-de/dev.ende.src", nlp_en, 'en')
de_val_mt = get_embeddings("../en-de/dev.ende.mt", nlp_de, 'de')
f_val_scores = open("../en-de/dev.ende.scores", 'r')
de_val_scores = f_val_scores.readlines()

# Print shapes
print(f"Training mt: {len(de_train_mt)} Training src: {len(de_train_src)}")
print()
print(f"Validation mt: {len(de_val_mt)} Validation src: {len(de_val_src)}")

# Put the features into a list
X_train = [np.array(de_train_src), np.array(de_train_mt)]
X_train_de = np.array(X_train).transpose()

X_val = [np.array(de_val_src), np.array(de_val_mt)]
X_val_de = np.array(X_val).transpose()

# Scores
train_scores = np.array(de_train_scores).astype(float)
y_train_de = train_scores

val_scores = np.array(de_val_scores).astype(float)
y_val_de = val_scores

# Call SVR
svr(X_train=X_train_de, y_train=y_train_de, X_val=X_val_de, y_val=y_val_de)

# Random Forest
rf = RandomForestRegressor(n_estimators=1000, random_state=666)
rf.fit(X_train_de, y_train_de)
predictions = rf.predict(X_val_de)
pearson = pearsonr(y_val_de, predictions)
print('RMSE:', rmse(predictions, y_val_de))
print(f"Pearson {pearson[0]}")


# EN-DE
de_test_src = get_embeddings("../en-de/test.ende.src", nlp_en, 'en')
de_test_mt = get_embeddings("../en-de/test.ende.mt", nlp_de, 'de')

X = [np.array(de_test_src), np.array(de_test_mt)]
X_test = np.array(X).transpose()

# Predict
clf_de = SVR(kernel='rbf')
clf_de.fit(X_train_de, y_train_de)

predictions_de = clf_de.predict(X_val_de)
write_scores("SVR", predictions_de)


'''
RMSE: 0.8813480780424763 Pearson 0.05429497493902449

poly
RMSE: 0.8815553219336182 Pearson 0.051014557060912215

rbf
RMSE: 0.8813934498530093 Pearson 0.054406144826044285

sigmoid
RMSE: 0.8813995524189149 Pearson 0.054855004379473174

RMSE: 0.9385994376085562
Pearson -0.03113502391691128'''
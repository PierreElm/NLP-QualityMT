# Requirements

Run the following command to install the Python 3 libraries needed to run the project:

```bash
pip install -r requirements.txt
```


Also, run the following commands so the Spacy Vocabulary
with the GloVe and Bert Embedding can be loaded:

```bash
python -m spacy download en_core_web_md
python -m spacy link en_core_web_md en300
python -m spacy download de_core_news_md
python -m spacy link de_core_news_md de300
python -m spacy download de_trf_bertbasecased_lg
python -m spacy download en_trf_xlnetbasecased_lg
```

Lastly, be sure to have an `en-de/` directory in the same level as the `src/` directory, 
with the training set, validation set and test set.

# Executing the files

The files can be excuted by running, for instance, the following:
```bash
python BertMLP.py
```

Make sure to be inside the `src/` directory when executing the previous command.

# Implementations

## IndicesRepresentation.py

This files implements the indices representations of each sentence.
Pre-processing parameters can be changed easily in the function call ``tokenize_corpuses``.
It allows us to try different pre-processing techniques without having to write new code.

## BagOfWords.py

This file implements the bag of words idea, the input given to the model is a vector of the size of the vocabulary build 
from each training files concatenated.
Multiple techniques to build the vocabulary can be used with this class. This
is also the case for training. Here is a list of the possible options that it provides:

1. Vocabulary construction possibilities (parameter ```n_gram```):
   - ```unigram```: The vocabulary built takes only each word into account
   - ```bigram```: The vocabulary built takes into account each words and bi-gram present in the training set.
   - ```trigram```: Takes into account, uni-gram, bi-gram and tri-gram when building the vocabulary.

2. Training methods (Depending on the methods called on the instance)
   - SVR
   - Multi Layer Perceptron

Exemples are provided in BagOfWords.py at the end of the files in comments.

## GloveCNN.py

In this file, we try to do regression using embedding on each word and different channels for each language.
Therefore, we use CNN with input of size: ```(batch_size, nb_channels, max_length, embedding_size)``` 
- _batch_size_: Size of each batch
- _nb_channels_: Number of languages (in our case 2) * number of embedding representation 
(For example if we use GloVe and FastText the number of channels will be 4)
- _max_length_: The length of the longest sentence of both corpus
- embedding_size_: The size of the embedding representation (300 for Glove)

## GloveMLP.py

In this file, we try to do regression using embeddings averaged for each word in a sentence.
The vectors of each sentence are concatenated into one vector.
Therefore, we use MLP with input of size: ```(batch_size, 2*embedding_size)``` 
- _batch_size_: Size of each batch
- embedding_size_: The size of the embedding representation (300 for Glove)


## GloveArc_I.py

In this file, we try to do regression using embedding on each word for each language. However the two 
languages will be indepedent inputs to the network.
Therefore, we use an 1d-CNN with input of size: ```(batch_size, 2*max_length, embedding_size)``` 
- _batch_size_: Size of each batch
- _max_length_: The length of the longest sentence of both corpus. The input here is 2*max_length
since the matrices of the two sentences are getting stacked on this dimension. However 
they are decoupled before forwarding and each matrix is forwared into a differen 1-dimensional
CNN.
- embedding_size_: The size of the embedding representation (300 for Glove)

## BertMLP.py

In this file, we try to do regression using embeddings averaged for each word in a sentence.
The vectors of each sentence are concatenated into one vector.

Therefore, we use MLP with input of size: ```(batch_size, 2*embedding_size)``` 
- _batch_size_: Size of each batch
- embedding_size_: The size of the embedding representation (768 for Bert)

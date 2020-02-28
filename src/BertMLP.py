from NN import *
import Helpers as hlp
import Embeddings as embs
from torch.optim import Adam
import torch.nn as nn
import torch
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from sklearn.metrics import make_scorer


USE_GPU = False

# if cross_validation is true it preform cross validation
# to calculate the error. Otherwise it calculate the error
# on the validation set.
cross_validation = False
# If true it creates a final model for submission on codalab
create_final_model = False

dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


input_dimension = 768*2  # input dimension is 768*2 since we stack the Bert embeddings as a single vector
# best hyperparameters selected by randomised search
epochs = 50
batch_size = 256
lr = 0.003491300804442097
weight_decay = 0.0010086745221866034

# Parameters for the class that creates embeddings.
# Here we specify that the Bert embedding are created,
# no stop words are removed and the case remains the same.
reader_params = {
    'bert': True,
    'lower': False,
    'stop_words': False,
}

# Read the training, validation and test data.
# Depending on the flags, the embeddings may not be created but written from a file. This is done to
# speedup the process when we want to try different architectures.
X_train_de, y_train_de, X_val_de, y_val_de, X_test = embs.get_data(embedding_class=embs.PretrainedEmbeddings,
                                                                   params=reader_params,
                                                                   read_embeddings_from_file=False,
                                                                   write_to_file=False,
                                                                   dtype=dtype)

# Create the Bert model
model = BertMLP(input_dimension)

if cross_validation:
    net = NeuralNetRegressor(model,
                             max_epochs=50,
                             lr=lr,
                             optimizer__weight_decay=weight_decay,
                             optimizer=torch.optim.Adam,
                             batch_size=256,
                             criterion=torch.nn.MSELoss,
                             )

    # Those are the parameters of the randomised search
    # params = {
    #     'lr': stats.uniform(0.00005, 0.01),
    #     'max_epochs': [50],
    #     'batch_size': [256, 512, 700, 1000],
    #     'optimizer__weight_decay': stats.uniform(0, 0.02),
    # }

    # Parameters selected by previous randomised search
    params = {
        'lr': [0.003491300804442097],
        'max_epochs': [50],
        'batch_size': [256],
        'optimizer__weight_decay': [0.0010086745221866034],
    }

    def pearson_error(y, preds):
        pearson_err = pearsonr(preds.squeeze(1), y.squeeze(1))[0]
        print(pearson_err)
        return pearson_err

    my_func = make_scorer(pearson_error, greater_is_better=True)

    # Performs randomised search with 7-fold cross validation
    rs = RandomizedSearchCV(net, params, refit=True, cv=5, scoring=my_func, n_iter=50, verbose=5)

    X_full_de = torch.cat((X_train_de, X_val_de), dim=0)
    y_full_de = torch.cat((y_train_de, y_val_de), dim=0)

    print(y_train_de.shape)
    rs.fit(X_full_de.float().detach().numpy(), y_full_de.unsqueeze(1).float().detach().numpy())

    print('Params: ', rs.best_params_)
    print('Score: ', rs.best_score_)

    """
    Params:  {'max_epochs': 15, 'lr': 0.0001, 'batch_size': 512}
    Score:  0.08445666054908306
    """
else:
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_nn(model, X_train_de, y_train_de, X_val_de, y_val_de, optimizer, nn.MSELoss(),
             device, batch_size=batch_size, epochs=epochs, dtype=dtype)

    X_val_de = X_val_de.to(device=device, dtype=dtype)
    model.eval()
    predictions = model(X_val_de.float()).squeeze(1).detach().numpy()
    pearson = pearsonr(y_val_de.detach().numpy(), predictions)
    print('RMSE:', rmse(predictions, y_val_de.detach().numpy()))
    print(f"Pearson {pearson[0]}")

    print("Printing training on test set plus validations set")

    if create_final_model:
        # Combines both the training set and the validation dataset to serve as the training set for the model
        # we submit on codalab.
        X_full_de = torch.cat((X_train_de, X_val_de), dim=0)
        y_full_de = torch.cat((y_train_de, y_val_de), dim=0)
        print(X_full_de.shape, y_full_de.shape)

        model = BertMLP(input_dimension)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_nn(model, X_full_de, y_full_de, X_val_de, y_val_de, optimizer, nn.MSELoss(),
                 device, batch_size=batch_size, epochs=epochs, dtype=dtype)
        X_test = X_test.to(device=device, dtype=dtype)
        model.eval()
        predictions = model(X_test.float()).squeeze(1).detach().numpy()
        hlp.write_scores("embeddigns_nn", predictions, filename="../predictions/bert_embedding_nn.txt")


from NN import *
import Helpers as hlp
import Embeddings as embs
from torch.optim import Adam
import torch.nn as nn
import torch
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetRegressor
from sklearn.metrics import make_scorer



USE_GPU = True

# if cross_validation is true it preform cross validation
# to calculate the error. Otherwise it calculate the error
# on the validation set.
cross_validation = False
# If true it creates a final model for submission on codalab
create_final_model = True

dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Read the training, validation and test data.
# Depending on the flags, the embeddings may not be created but written from a file. This is done to
# speedup the process when we want to try different architectures.
X_train_de, y_train_de, X_val_de, y_val_de, X_test = embs.get_data(embedding_class=embs.PretrainedEmbeddings,
                                                                   read_embeddings_from_file=False,
                                                                   write_to_file=False,
                                                                   dtype=dtype)


input_dimension = 600  # input dimension is 600 since we stack the embeddings as a single vector

# Hyperparameters
epochs = 15
batch_size = 700
weight_decay = 0.010784494606556184
lr = 0.006456371077398479


model = MLP(input_dimension)

if cross_validation:
    net = NeuralNetRegressor(model,
                             max_epochs=50,
                             lr=lr,
                             optimizer__weight_decay=weight_decay,
                             optimizer=torch.optim.Adam,
                             batch_size=256,
                             criterion=torch.nn.MSELoss,
                             )

    # Parameters for randomised search
    # params = {
    #     'lr': stats.uniform(0.00005, 0.01),
    #     'max_epochs': [15, 20, 30],
    #     'batch_size': [128, 256, 512, 700, 1000],
    #     'optimizer__weight_decay': stats.uniform(0, 0.1),
    # }

    # best parameters
    params = {
        'lr': [0.006456371077398479],
        'max_epochs': [15],
        'batch_size': [700],
        'optimizer__weight_decay': [0.010784494606556184],
    }

    def pearson_error(y, preds):
        pearson_err = pearsonr(preds.squeeze(1), y.squeeze(1))[0]
        print(pearson_err)
        return pearson_err

    my_func = make_scorer(pearson_error, greater_is_better=True)

    # Performs randomised search with 7-fold cross validation
    rs = RandomizedSearchCV(net, params, refit=True, cv=7, scoring=my_func, n_iter=200, verbose=5)

    rs.fit(X_train_de.float().detach().numpy(), y_train_de.unsqueeze(1).float().detach().numpy())

    print('Params: ', rs.best_params_)
    print('Score: ', rs.best_score_)

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

        model = MLP(input_dimension)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_nn(model, X_full_de, y_full_de, X_val_de, y_val_de, optimizer, nn.MSELoss(),
                 device, batch_size=batch_size, epochs=epochs, dtype=dtype)
        X_test = X_test.to(device=device, dtype=dtype)
        model.eval()
        predictions = model(X_test.float()).squeeze(1).detach().numpy()
        hlp.write_scores("embeddigns_nn", predictions, filename="../predictions/embedding_nn.txt")

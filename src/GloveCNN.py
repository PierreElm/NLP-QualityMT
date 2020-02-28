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
X_train_de, y_train_de, X_val_de, y_val_de, X_test = embs.get_data(embedding_class=embs.NgramEmbedding,
                                                                   read_embeddings_from_file=False,
                                                                   write_to_file=False,
                                                                   dtype=dtype, stack_as_channels=True)

# Parameters
N, nb_channels, sentence_size, embeddings_size = X_train_de.size()
weight_decay = 0.001
lr = 0.001
epochs = 10
batch_size = 256

# Create model
model = CNN(sentence_size, in_channels=nb_channels, embedding_size=embeddings_size, c_out=2, bias=True)

parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters is: {}".format(parameters))

if cross_validation:
    # skorch wrapper, needed to for sklearn randomised search on a pytorch model
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
    #     'lr': [5e-4, 1e-3],
    #     'max_epochs': [10, 20, 30],
    #     'batch_size': [128, 256, 512],
    #     'optimizer__weight_decay': [1e-2, 1e-3],
    # }

    # Parameters determined by cross validation
    params = {
        'lr': [1e-3],
        'max_epochs': [10],
        'batch_size': [256],
        'optimizer__weight_decay': [1e-3],
    }

    def pearson_error(y, preds):
        pearson_err = pearsonr(preds.squeeze(1), y.squeeze(1))[0]
        return pearson_err

    my_func = make_scorer(pearson_error, greater_is_better=True)

    # Performs randomised search with 7-fold cross validation
    rs = RandomizedSearchCV(net, params, refit=True, cv=7, scoring=my_func, n_iter=1, verbose=5)

    print(y_train_de.shape)
    rs.fit(X_train_de.float().detach().numpy(), y_train_de.unsqueeze(1).float().detach().numpy())

    print('Params: ', rs.best_params_)
    print('Score: ', rs.best_score_)

else:
    model.to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_nn(model, X_train_de, y_train_de, X_val_de, y_val_de, optimizer, nn.MSELoss(), device,
             batch_size=batch_size, epochs=epochs, dtype=dtype)

    # Validation
    X_val_de = X_val_de.to(device=device, dtype=dtype)
    model.eval()
    predictions = model(X_val_de.float()).squeeze(1).cpu().detach().numpy()
    pearson = pearsonr(y_val_de.cpu().detach().numpy(), predictions)
    print('RMSE:', rmse(predictions, y_val_de.cpu().detach().numpy()))
    print(f"Pearson {pearson[0]}")

    if create_final_model:
        # Combines both the training set and the validation dataset to serve as the training set for the model
        # we submit on codalab.
        X_full_de = torch.cat((X_train_de, X_val_de.cpu()), dim=0)
        y_full_de = torch.cat((y_train_de, y_val_de), dim=0)
        print(X_full_de.shape, y_full_de.shape)

        model = CNN(sentence_size, in_channels=nb_channels, embedding_size=embeddings_size, c_out=2, bias=True).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_nn(model, X_full_de, y_full_de, X_val_de, y_val_de, optimizer, nn.MSELoss(),
                 device, batch_size=batch_size, epochs=epochs, dtype=dtype)

        X_test = X_test.to(device=device, dtype=dtype)
        model.eval()
        predictions = model(X_test.float()).squeeze(1).cpu().detach().numpy()
        hlp.write_scores("embedding_cnn", predictions, filename="../predictions/embedding_cnn.txt")


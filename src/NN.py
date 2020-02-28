from torch import nn
from Helpers import rmse
import torch
from scipy.stats.stats import pearsonr
import math


# MLP used with Glove Embeddings
class MLP(nn.Module):
    def __init__(self, input_dimension):
        # Call parent
        super(MLP, self).__init__()

        # Architecture
        self.fc1 = nn.Linear(in_features=input_dimension, out_features=500)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.out = nn.Linear(in_features=500, out_features=1)

    def forward(self, x):
        """ Forward pass """
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.out(x)
        return x


# MLP used with Bert Embeddings
class BertMLP(nn.Module):
    def __init__(self, input_dimension, dim1=500, dim2=150):
        # Call parent
        super(BertMLP, self).__init__()
        # Architecture
        self.fc1 = nn.Linear(in_features=input_dimension, out_features=dim1)
        self.relu1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=dim1, out_features=dim2)
        self.relu2 = nn.LeakyReLU()
        self.drop2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(in_features=dim2, out_features=1)

    def forward(self, x):
        """ Forward pass """
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.out(x)
        return x


# CNN used with Glove Embeddings
class CNN(nn.Module):
    def __init__(self, sentence_size, in_channels=2, embedding_size=300, c_out=2, bias=True):
        # Call parent
        super(CNN, self).__init__()

        self.channels_out = c_out
        self.s_size = sentence_size
        # Architecture
        # Region size of 4
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c_out, kernel_size=(4, embedding_size), bias=bias),
            nn.ReLU(),
            nn.MaxPool2d((sentence_size - 3, 1))
        )
        # Region size of 3
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c_out, kernel_size=(3, embedding_size), bias=bias),
            nn.ReLU(),
            nn.MaxPool2d((sentence_size - 2, 1))
        )
        # Region size of 2
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c_out, kernel_size=(2, embedding_size), bias=bias),
            nn.ReLU(),
            nn.MaxPool2d((sentence_size - 1, 1))
        )
        # Regression
        self.fcl = nn.Linear(in_features=self.channels_out * 3, out_features=1)

    def forward(self, x):
        x4 = self.conv1_4(x).view(-1, self.channels_out)
        x3 = self.conv1_3(x).view(-1, self.channels_out)
        x2 = self.conv1_2(x).view(-1, self.channels_out)
        # Concatenate filters to form single feature vector
        x = torch.cat((x4, x3, x2), dim=1)
        x = self.fcl(x)
        return x


# Used with Glove Embeddings
class ARC_I(nn.Module):
    def __init__(self, sentence_size=23, out_channels=8, embedding_size=300):
        # Call parent
        super(ARC_I, self).__init__()

        self.sentence_size = sentence_size
        self.out_channels = out_channels
        self.kernel_size = 5
        # size of output after convolution
        self.l_out = 23 - (self.kernel_size-1) - 1 + 1
        self.max_kernel = 2
        self.l_out = math.floor((self.l_out - (self.max_kernel-1) - 1)/self.max_kernel + 1)
        print(self.l_out)
        self.conv_en = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      bias=True),
            nn.ReLU(),
            # nn.BatchNorm1d(self.out_channels),
            nn.Dropout2d(p=0.2),
            nn.MaxPool1d(self.max_kernel)
        )

        self.conv_de = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      bias=True),
            nn.ReLU(),
            # nn.BatchNorm1d(self.out_channels),
            nn.Dropout2d(p=0.2),
            nn.MaxPool1d(self.max_kernel)
        )

        # Regression
        self.fcl = nn.Linear(in_features=2*self.out_channels*self.l_out,
                             out_features=1)

    def forward(self, x):
        x_en = self.conv_en(x[:, :, :self.sentence_size]).view(-1, self.out_channels*self.l_out)
        x_de = self.conv_de(x[:, :, self.sentence_size:]).view(-1, self.out_channels*self.l_out)
        # Concatenate filters to form single feature vector
        x = torch.cat((x_en, x_de), dim=1)
        x = (self.fcl(x))
        return x


def train_nn(model, input, target, input_val, target_val, optimizer, loss_fn, device, batch_size=64,
             epochs=10, dtype=torch.float64, input_2=None, input_2_val=None):
    """
    The basing neural network training loop that works for all our networks
    :param model: the model that we want to train
    :param input: the training features
    :param target: the training target values
    :param input_val: the validation features, used for providing performance metrics during training
    :param target_val: the validation target values
    :param optimizer: the optimizer used for training
    :param loss_fn: the loss function that we minimze
    :param device: either cuda or gpu
    :param batch_size: the batch size for SGD
    :param epochs: the number of epochs to train
    :param dtype: the datatypes of the tensors we are using
    :param input_2: used for the arc1 input
    :param input_2_val:  used for the arc1 validation
    :return:
    """
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(input.size()[0])
        for i in range(0, input.size()[0], batch_size):
            model.train()

            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = input[indices], target[indices]
            batch_x = batch_x.to(device=device, dtype=dtype)
            batch_y = batch_y.to(device=device, dtype=dtype)
            # in case you wanted a semi-full example
            if input_2 is not None:
                batch_x_2 = input_2[indices].to(device=device, dtype=dtype)
                preds = model(batch_x.float(), batch_x_2.float()).squeeze(1)
            else:
                preds = model(batch_x.float()).squeeze(1)
            loss = loss_fn(preds, batch_y.float())

            # Compute the loss
            train_loss = loss.item()
            # calculate the gradient of each parameter
            loss.backward()
            optimizer.step()

        # every epoch calculate the pearson correlation on the training dataset
        model.eval()
        if input_2 is not None:
            predictions = model(input.to(device).float(), input_2.to(device).float()).squeeze(1).cpu().detach().numpy()
        else:
            predictions = model(input.to(device).float()).squeeze(1).cpu().detach().numpy()
        y = target.detach().numpy()
        pearson = pearsonr(y, predictions)

        # every epoch calculate the pearson correlation on the validation dataset
        if input_2_val is not None:
            predictions_val = model(input_val.to(device).float(), input_2_val.to(device).float()).squeeze(1).cpu().detach().numpy()
        else:
            predictions_val = model(input_val.to(device).float()).squeeze(1).cpu().detach().numpy()
        y_val = target_val.detach().numpy()
        pearson_val = pearsonr(y_val, predictions_val)
        rmse_val = rmse(predictions_val, y_val)
        print(f'| Epoch: {epoch:02} | Train Loss: {train_loss:.3f}| Pearson Train: {pearson[0]:.6f} |  Pearson Val: {pearson_val[0]:.6f}'
              f'| RMSE Val  {rmse_val:.3f}')

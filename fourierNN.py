"""Simple file to show images from DataSet"""
import torch
import torch.fft
from src.main_ import finalize, get_dataloader, initialize
from src.utils.config import Conf
from src.utils.enums import TDataUse
from src.utils.log import log
from src.utils.misc_func import strs_to_classes
import matplotlib.pyplot as plt
import trainNet as tNet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import embed
import numpy as np

USE_GPU = True
DATA_LOADER_WORKERS = 4#If there are issues with the data loader set this to 1 and try this agian


dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def process(x):
    x = torch.flip(x,[2,3])
    x = torch.fft.fftn(x, s=(224,224)).abs() # move to device, e.g. GPU
    torch.cumsum(x, dim=2, out=x)
    torch.cumsum(x, dim=3, out=x)
    x = x/10000
    #embed()
    ind = torch.arange(0,200, device=device)
    xVec = x[:,1,ind,ind]
    
    #xi = torch.fft.fftn(x, s=(224,224)).imag # move to device, e.g. GPU
    #x = torch.cat((xr,xi),1)
    return xVec
def check_accuracy(loader, model):
  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
          
            xs = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            x = process(xs)
            
            y = y.to(device=device, dtype=torch.long)

            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            #embed()
            
            break #only go 1 iteration
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_model(model, optimizer, loader_train, loader_val, epochs, evalIts):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """


    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            
            xs = x.to(device=device, dtype=dtype)
            x = process(xs)
            
            y = y.to(device=device, dtype=torch.long)

            yLog = torch.zeros(y.shape[0], 2, device = device)
            yLog[range(yLog.shape[0]), y]=1
            #embed()
            scores = flatten(model(x))
            # scores = model(x)
 
            loss = F.binary_cross_entropy_with_logits(scores, yLog)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % evalIts == 0:
                print('Iteration %d, loss = %.4f' % (e, loss.item()))
                check_accuracy(loader_val, model)

           
         


def main():
    print(device)
    #main.main()
    filters_as_str = None  # If left as non defaults will be used
    if filters_as_str is None:
        filters_as_str = Conf.RunParams.MODEL_TRAIN_DEFAULTS['filters']
    filters = strs_to_classes(filters_as_str)
    #log(f'Going to test: {filters_as_str}')
    loader_train,loader_val,loader_test = get_dataloader(filters)
  

    hidden_layer_size = 100
    
    x = nn.Linear(200, hidden_layer_size)
    y = nn.Linear(hidden_layer_size,2)
    
    x.weights = nn.Parameter(tNet.random_weight((200,hidden_layer_size)))
    y.weights = nn.Parameter(tNet.random_weight((hidden_layer_size,2)))
    model = nn.Sequential(
        Flatten(),
        x,
        nn.ReLU(),
        y
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)
    model = model.to(device)
    train_model(model, optimizer, loader_train=loader_train, loader_val=loader_val, epochs = 1, evalIts = 10)
    #check_accuracy(loader_test,model)

   

if __name__ == '__main__':
    main()

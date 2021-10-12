import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import torchvision.models as models
import pathlib
from pathlib import Path
from torch.utils.data import random_split
from convNeuralNetwork import ConvNeuralNetwork
from mnistImageDataset import MnistImageDataset
from mnistTestImageDataset import MnistTestImageDataset
from PIL import Image
from matplotlib import pyplot as plt
import sys


# Launch arguments
loadModel = False
if len(sys.argv) > 1 and sys.argv[1] == "load":
    loadModel = True

# define training hyperparameters
INIT_LR = 5e-4
BATCH_SIZE = 64
EPOCHS = 50
WEIGHT_DECAY = 1e-6

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT


def train_loop(dataloader, model, loss_fn, optimizer):
    # set the model in training mode
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # initialize the total training and validation loss
    totalTrainLoss = 0
    
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    
    for (X, y) in dataloader:
        # load data on GPU
        (X, y) = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()

    totalTrainLoss /= num_batches
    trainCorrect /= size
    print(f"Train Error: \n Accuracy: {(100*trainCorrect):>0.3f}%, Avg loss: {totalTrainLoss:>8f} \n")
    return 100*trainCorrect


def validation_loop(dataloader, model, loss_fn):

    model.eval()
    totalValLoss = 0
    valCorrect = 0

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    

    with torch.no_grad():

        for (X, y) in dataloader:
            # load data on GPU
            X, y = X.to(device), y.to(device)
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            totalValLoss += loss
            valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        totalValLoss /= num_batches
        valCorrect /= size
        print(f"Validation Error: \n Accuracy: {(100*valCorrect):>0.3f}%, Avg loss: {totalValLoss:>8f} \n")
        return 100*valCorrect



def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    results = []
    
    with torch.no_grad():
        for X in dataloader:
             # load data on GPU
            X= X.to(device)
            pred = model(X)
            correctClasses = pred.argmax(1).cpu().detach().numpy()
            results = np.append(results, correctClasses)
        
        ids = np.arange(1, size+1)
        result_df = pd.DataFrame({'ImageId': ids, 'Label':results.astype(int)})
        print(result_df)
        result_df.to_csv(f"{localpath}/submission.csv", index = False)

# my_model = Path(f"{localpath}/model.pth")
# if my_model.is_file():
#    print("Model already exists !!!!")
#    model = torch.load('model.pth')
# else :

localpath = pathlib.Path().resolve()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device} device")

# Import training data
df_train = pd.read_csv(f"{localpath}/train.csv")
df_train_features = df_train.drop('label', inplace=False, axis=1).to_numpy()
df_train_features = df_train_features.reshape(-1, 28, 28, 1).astype(np.uint8)
print(df_train_features.shape)
print(df_train_features.dtype)
#plt.imshow(df_train_features[6], cmap='gray')
#plt.show()



df_train_labels = df_train['label'].to_numpy().reshape(-1)
print(df_train_labels.shape)
print(df_train_labels[10])


# Create training Dataset
dataset_train = MnistImageDataset(
    df_train_features, 
    df_train_labels,
    transform=ToTensor(), 
    target_transform= None#Lambda(lambda y: torch.zeros(10, dtype=torch.long).scatter_(0, torch.tensor(y), value=1))
    )

# calculate the train/validation split
def trainValidationSplit(dataset_train) : 

    print("[INFO] generating the train/validation split...")
    numTrainSamples = int(len(dataset_train) * TRAIN_SPLIT)
    numValSamples = int(len(dataset_train) * VAL_SPLIT)
    (trainData, valData) = random_split(dataset_train,
        [numTrainSamples, numValSamples])

    return DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True), DataLoader(valData, batch_size=BATCH_SIZE, shuffle=False)


dataloader_train, dataloader_val = trainValidationSplit(dataset_train)

# Import testing data
df_test = pd.read_csv(f"{localpath}/test.csv")
df_test_formated = df_test.to_numpy().reshape(-1, 28, 28, 1).astype(np.uint8)
testDataset = MnistTestImageDataset(df_test_formated, transform=ToTensor())
print("test dataset length")
print(len(testDataset))
dataloader_test = DataLoader(testDataset, batch_size=BATCH_SIZE)

# Loading model or creating it
my_model = Path(f"{localpath}/modelMnist.pth")
if my_model.is_file() and loadModel:
    print("Loading model from modelMnist.pth")
    model = torch.load('modelMnist.pth')
else :
    print("Creating Model")
    model = ConvNeuralNetwork(numChannels=1, classes=10).to(device)

# Initialize the loss function
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=1e-5)


start = time.time()

bestValAccuracy = 0.0
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    trainAccuracy = train_loop(dataloader_train, model, loss_fn, optimizer)
    valAccuracy = validation_loop(dataloader_val, model, loss_fn)
    if (valAccuracy > bestValAccuracy) :
        #Saving model
        print("Saving model...")
        torch.save(model, 'modelMnist.pth')
        bestValAccuracy = valAccuracy
    
print("Done!")
end = time.time()
print(f"Executed in {end - start} seconds.")
my_model = Path(f"{localpath}/modelMnist.pth")
if my_model.is_file():
    print("Loading best model for testing set")
    model = torch.load('modelMnist.pth')
test_loop(dataloader_test, model)



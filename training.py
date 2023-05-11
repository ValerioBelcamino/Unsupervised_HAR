import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset  # For data loading and batching
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAccuracy
from torchinfo import summary

from models import LSTMMultiClass, TransformerClassifier, LSTMBinary, CNN_1D, CNN_1D_multihead

training = True

balanced_dataset = False
binary_classification = False

current_action = 'ASSEMBLY1'

n_folds = 10
n_perm = 64 

def oneVsAll(labels, int_labels, label):
    '''Converts a multi-class problem into a binary problem by setting all labels'''
    unique_labels = np.unique(labels)
    print(unique_labels)
    int_label = np.argwhere(unique_labels == label)[0][0]
    mask = int_labels == int_label
    int_labels[mask] = 0
    int_labels[~mask] = 1
    return int_labels

def ignore_gyro_features(dataset):
    '''Excludes gyroscope data from the dataset'''
    mask = np.ones((24), dtype=bool)
    mask[[3,4,5,9,10,11,15,16,17,21,22,23]] = False
    dataset = dataset[:, :, mask]
    return dataset



def get_accuracy(pred, test):
    '''Returns the accuracy of the model on the multiclass problem'''
    correct = 0
    wrong = 0
    for p, t in zip(torch.argmax(pred,1), test):
        if p == t:
            correct+=1
        else:
            wrong+=1
    return (correct/test.shape[0])*100

def normalize(data):
    '''Normalizes the data by dividing each feature by its maximum value'''
    maxes = np.amax(data, axis=(0,1))
    print(maxes)
    # mins = np.amin(data, axis=(0,1))
    # return (2*(data-mins)/(maxes-mins))-1
    return data/maxes

def full_scale_normalize(data):
    if data.shape[-1] == 24:
        acceleration_idxs = [0,1,2,6,7,8,12,13,14,18,19,20]
        gyroscope_idxs = [3,4,5,9,10,11,15,16,17,21,22,23]

        # 1g equals 8192. The full range is 2g
        data[:,:,acceleration_idxs] = data[:,:,acceleration_idxs] / 16384.0
        data[:,:,gyroscope_idxs] = data[:,:,gyroscope_idxs] / 100.0

    else: 
        data = data / 16384.0

    return data

def add_feature_profiles(dataset):
    '''Adds the module of the 3D vector of each feature to the dataset'''
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], -1, 3)
    module = np.sqrt(np.sum(dataset**2, axis=3))
    dataset = np.concatenate((dataset, module[..., None]), axis=3)
    return dataset.reshape(dataset.shape[0], dataset.shape[1], -1)

def add_precision_recall(matrix, accuracy):
    tmp = np.zeros((matrix.shape[0]+1, matrix.shape[1]+1))

    tmp[-1,-1] = accuracy
    tmp[:-1,:-1] += matrix

    '''PRECISION'''
    for i in range(matrix.shape[0]):
        tmp[i, -1] = tmp[i,i] / np.sum(tmp, axis=1)[i] * 100
    '''RECALL'''
    for j in range(matrix.shape[1]):
        tmp[-1, j] = tmp[j,j] / np.sum(tmp, axis=0)[j] * 100
    
    return tmp 


def get_folds(n_folds, dataset, perms_x_seq):
    fold_len = int(dataset.shape[1]/n_folds)
    pad = int(fold_len * 0.05)

    permutation_matrix = np.zeros((n_perm, n_folds), dtype=int)
    np.random.seed(42)

    for i in range(n_perm):
        permutation_matrix[i] = np.random.permutation(n_folds)

    labels_new = np.zeros((dataset.shape[0] * perms_x_seq), dtype=int)
    dataset_new = np.zeros((dataset.shape[0] * perms_x_seq, dataset.shape[1] - (pad * n_folds), dataset.shape[2]))

    

    for i in range(dataset.shape[0]):
        for j in range(perms_x_seq):
            rand_perm_idx = np.random.randint(n_folds, size=1)
            dataset_new[i+j] = get_seq_folds(fold_len, pad, permutation_matrix[rand_perm_idx, :][0], dataset[i])
            labels_new[i+j] = rand_perm_idx
    return dataset_new, labels_new


def get_seq_folds(fold_len, pad, permutation, sequence):
    folds = []
    for i in permutation:
        folds.append(sequence[i*fold_len:(i+1)*fold_len-pad])

    return np.row_stack(folds)


def get_classification_tresholds(preds, labels):
    '''Returns the tresholds for each class'''

    preds = preds.cpu()
    labels = labels.cpu()

    preds = preds.detach().numpy()
    labels = labels.detach().numpy()

    thresholds = {}
    not_thresholds = {}

    unique_labels = np.unique(labels)

    for y_p, y_r in zip(preds, labels):
        if np.argmax(y_p) == y_r:
            for lab in unique_labels:
                if lab == y_r:
                    if y_r not in thresholds.keys():
                        thresholds[y_r] = [y_p[y_r]]
                    else:
                        thresholds[y_r].append(y_p[y_r])
                else:
                    if lab not in not_thresholds.keys():
                        not_thresholds[lab] = [y_p[lab]]
                    else:
                        not_thresholds[lab].append(y_p[lab])

    for key in thresholds:
        thresholds[key] = np.mean(thresholds[key])#*0.8
    for key in not_thresholds:
        not_thresholds[key] = (1-np.mean(not_thresholds[key])) *0.2

    print(f'{thresholds=}')
    print(f'{not_thresholds=}')

    return thresholds, not_thresholds


print("\n--- Data Loading ---")



print("\n--- Loading Unbalanced Dataset ---")
train_dataset = np.load('full_sequencies_train.npy').astype('float32')
test_dataset = np.load('full_sequencies_test.npy').astype('float32')


train_dataset, train_labels = get_folds(n_folds, train_dataset, 1)
test_dataset, test_labels = get_folds(n_folds, test_dataset, 1)



train_dataset = full_scale_normalize(train_dataset)
test_dataset = full_scale_normalize(test_dataset)  
print("\nSplitted dataset and labels: ")
print(f'\t{train_dataset.shape=}')
print(f'\t{test_dataset.shape=}')



print("\n--- Training ---")
# Set device to CUDA if available, otherwise use CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'\nSetting torch device to: {device=}')

x = torch.Tensor(train_dataset).to(device)
y = torch.Tensor(train_labels).squeeze().long().to(device)

x_test = torch.Tensor(test_dataset).to(device)
y_test = torch.Tensor(test_labels).squeeze().long().to(device)
print(f'\t{x.shape=}')
print(f'\t{y.shape=}')

# Define hyperparameters
input_dim = train_dataset[0].shape[-1]
hidden_dim = 256
n_layers = 2
if binary_classification:
    output_dim = 1
else:
    output_dim = n_perm
'''multihead cnn works best with 0.0005'''
'''singlehead cnn works best with 0.0001'''

lr = 0.0001
epochs = 200
batch_size = 32
dropout = 0.5
l2_lambda = 0.00005

# Set up early stopping
patience = 8
best_val_loss = float('inf')
counter = 0

print(f'\nHyperparameters: ')
print(f'\t{input_dim=}')
print(f'\t{hidden_dim=}')
print(f'\t{output_dim=}')
print(f'\t{lr=}')
print(f'\t{epochs=}')
print(f'\t{batch_size=}\n')
print(f'\t{l2_lambda=}\n')
print(f'\t{patience=}\n')

# Instantiate the model
if binary_classification:
    model = LSTMBinary(input_dim, hidden_dim, output_dim, n_layers, dropout).cuda()
else:
    # model = LSTMMultiClass(input_dim, hidden_dim, output_dim, n_layers, dropout).cuda()
    # model = CNN_1D(input_dim, output_dim, dropout).cuda()
    model = CNN_1D_multihead(input_dim, output_dim).cuda()
    # model = TransformerClassifier(input_dim, output_dim, hidden_dim, n_layers, nheads).cuda()
# print(f'{model}')
summary(model, input_size=(batch_size, train_dataset[0].shape[0], train_dataset[0].shape[1]))
# exit()
if binary_classification:
    criterion= nn.BCELoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

# Create a DataLoader for the input data and labels
print(f'{x.shape=}')
print(f'{y.shape=}')
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print(f'{x_train.shape=}')
print(f'{y_train.shape=}')
print(f'{x_val.shape=}')
print(f'{y_val.shape=}')
data_train = TensorDataset(x_train, y_train)
# data_val = TensorDataset(x_val, y_val)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
x_val = x_val.cuda()
y_val = y_val.cuda()

metric = BinaryAccuracy(device=torch.device('cuda'))



if training:
    for epoch in range(epochs):
        # At the beginning of each epoch, set your model to train mode
        model.train()
        
        # Loop through your training data
        for i, batch in enumerate(train_loader):
            # Unpack the batch
            batch_x, batch_y = batch
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            # print(f'{batch_x.device=}')
            # Reset the gradients to zero
            optimizer.zero_grad()
            
            # Pass the input sequence through the model and get the predicted output
            output = model(batch_x)
            
            # Calculate the loss between the predicted output and the true output

            if binary_classification:
                loss = criterion(output.squeeze(), batch_y.float())
            else:
                loss = criterion(output, batch_y)
            # Backpropagate the loss through the network and update the model parameters
            loss.backward()
            optimizer.step()

            del batch_x
            del batch_y
            del output
            torch.cuda.empty_cache()
        model.eval()

        y_pred = model(x_val)
        
        if binary_classification:
            val_loss = criterion(y_pred.squeeze(), y_val.float())
            metric.update(y_pred.squeeze(), y_val)
            acc = metric.compute()
        else:
            val_loss = criterion(y_pred, y_val)
            acc = get_accuracy(y_pred, y_val)

        del y_pred

        best = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = " -- Best Yet"
            if binary_classification:
                torch.save(model.state_dict(), f"binary_models/{current_action}.pth")
            else:
                torch.save(model.state_dict(), "best_model.pth")
            counter = 0
        else:
            counter += 1
        
        print(f"Epoch {epoch}: Training Loss = {loss.item():.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {acc:.2f}" + best)
        
        if counter == patience:
            print(f"Stopping training after {epoch} epochs due to no improvement in validation loss.")
            break

if binary_classification:
    model.load_state_dict(torch.load(f"binary_models/{current_action}.pth"))
else:
    model.load_state_dict(torch.load("best_model.pth"))
model.eval()
x_test = x_test.cuda()
y_pred = model(x_test)  
active_thr, inactive_thr = get_classification_tresholds(y_pred, y_test)

if binary_classification:
    metric = BinaryAccuracy(device=torch.device('cuda'))
    metric.update(y_pred.squeeze().cuda(), y_test.cuda())
    acc = metric.compute()
else:
    acc = get_accuracy(y_pred, y_test)
print("accuracy: ", acc)


# Build confusion matrix
if binary_classification:
    y_pred = y_pred>0.5
    y_pred = y_pred.cpu()
    cf_matrix = confusion_matrix(y_pred, y_test.cpu())
else:
    cf_matrix = confusion_matrix(torch.argmax(y_pred,1).cpu(), y_test.cpu())
    # cf_matrix = add_precision_recall(cf_matrix, acc)

print(cf_matrix)
print(np.sum(cf_matrix, axis=1)[:, None])
cf_matrix = np.around(add_precision_recall(cf_matrix / np.sum(cf_matrix, axis=1)[:, None] * 100, acc), decimals=1)

# df_cm = pd.DataFrame(cf_matrix, index = [i for i in unique_labels],
#                      columns = [i for i in unique_labels])
plt.figure(figsize = (12,7))
sn.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
if binary_classification:
    plt.savefig(f'binary_models/{current_action}_cf.png')
else:
    plt.savefig('output.png')

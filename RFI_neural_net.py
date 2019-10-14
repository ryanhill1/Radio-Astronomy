# Ryan Hill
# Cornell University, Dept. of Astronomy
# Program to categorize radio frequency interference
# via convolutional neural network

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

# MSE Loss
def MSE_pred_to_tensor(pred):
  t = [[0]*4]
  t[0][pred] = 1
  t = torch.tensor(t)
  return t

# Cross Entropy Loss
def CE_pred_to_tensor(pred):
  t = [pred]
  t = torch.tensor(t)
  return t

# Load Data
data = Path("/home/ella2/ryan/training_data/")

# Pair data with labels
def create_training_data():
  categories = ['llbb', 'llnb', 'slbb', 'slnb', 'znoise']
  training_data = []
  for category in categories:
    path = data / category
    class_num = categories.index(category)
    for plt_array in path.iterdir():
      plt_array = np.load(plt_array, allow_pickle=True)
      plt_array = torch.tensor(plt_array)
      plt_array = plt_array.unsqueeze(0)  # represents 1 color channel
      # append as tuple (image, label)
      training_data.append((plt_array, CE_pred_to_tensor(class_num)))
  return training_data

training_data = create_training_data()
random.shuffle(training_data)  # put training data in random order

# print(len(training_data))
# print((training_data[0][0]).size())

# Define the network
class Network(nn.Module):

  def __init__(self):
    super(Network, self).__init__()

    # input image channels, output channels, NxN square convolutional kernel
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6,
        kernel_size=7, stride=1, padding=0),
      nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=6, out_channels=12,
        kernel_size=7, stride=1, padding=0),
      nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
    # input features, output features
    self.fc1 = nn.Linear(in_features=12*26*26,
      out_features=5)  # linear layer
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, x):
    # input layer implicit (identity)

    # hidden convolutional layers
    x = self.layer1(x)
    x = self.layer2(x)
    # print(x.shape)

    # flatten matrix
    x = x.reshape(x.size(0), -1)

    # linear layer
    x = self.fc1(x)

    # softmax / output
    # x = self.softmax(x) # don't use softmax with cross-entropy

    return x

network = Network()  # create an instance

# Display layout of network
# for name, param in network.named_parameters():
#     print(name, '\t\t', param.shape)

# Define a loss function and an optimizer
# criterion = nn.MSELoss(reduction='sum')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=1e-5)
# optimizer = optim.SGD(network.parameters(), lr=1e-5, momentum=0.9)

# list to hold loss at each epoch for graphing
adam_loss = []

# Train the network
print()
print("******************************")
print("START TRAINING")
print()
network = network.float()
start = time.time()
print("starting timer")

# initialize at epoch 1
def train(epoch):
  running_loss = 0.0
  for i in range(len(training_data)):
    # get the inputs; data is a list of [inputs, labels]
    data_file = training_data[i]
    inputs, labels = data_file

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    # loss = criterion(outputs, labels.float()) # uncomment for MSE
    loss = criterion(outputs, labels)
    # print(loss)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 10000 == 9999:  # print every 10000 mini-batches (1 epoch)
      print('epoch %d, loss: %.3f' %
        (epoch, running_loss / 10000))
      adam_loss.append(running_loss / 10000)
      epoch += 1
      if epoch >= 26:
      # if running_loss <= (0.155 * 10000):
        print('Finished Training')
        return
  train(epoch)

train(1)
end = time.time()
print("total time = ", end - start)  # time in seconds
print()
print("END TRAINING")
print("******************************")
print()

# Load test data
path = Path("/home/ella2/ryan/test_data_snr/")

def create_test_data():
  test_data = []
  for plt_cat_snr in path.iterdir():
    plt_cat_snr = np.load(plt_cat_snr, allow_pickle=True)
    plt, category, snr = plt_cat_snr[0], plt_cat_snr[1], plt_cat_snr[2]
    plt, category, snr = torch.tensor(plt), torch.tensor(category), torch.tensor(snr)
    plt = plt.unsqueeze(0)  # represents 1 color channel
    test_data.append((plt, category, snr))  # append as tuple (image, label)
  return test_data  # category tensor shape only compatable with Cross Entropy Loss

test_data = create_test_data()
# print(len(test_data))

# See how the network performs on the whole dataset
# Number of correct classifications
def get_num_correct(preds, labels):
  return preds.argmax(dim=1).eq(labels).sum().item()

print()
print("******************************")
print("STATISTIC FOR TEST DATA")
print()

correct = 0
total = 0
network = network.float()
with torch.no_grad():
  for i in range(len(test_data)):
    data_file = test_data[i]
    inputs, labels, _ = data_file

    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 500 test images: %d %%' % (
    100 * correct / total))

# # Graph the loss
# plt.plot(adam_loss, '-b', label='Adam')
# # plt.plot(sgd_loss,'-r',label='SGD')
# plt.xlabel('Epoch number', fontsize=14)
# plt.ylabel('Mean loss', fontsize=14)
# plt.title('Training 2000/class to loss = 0.05')
# plt.legend(loc='best', numpoints=1)
# plt.ylim(0., 1.0)
# plt.savefig('2000_to_loss05.png')
# plt.show()
# plt.close()

# which classes performed well, which did not perform well
classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
with torch.no_grad():
  for i in range(len(test_data)):
    data_file = test_data[i]
    inputs, labels, _ = data_file

    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).item()
    for i in range(4):
      # print(labels)
      label = labels.item()
      class_correct[label] += c
      class_total[label] += 1

for i in range(5):
  print('Accuracy of %5s : %2d %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))

# identify misclassifications
def most_common(lst):
  classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
  if lst == []: 
    return 'none'
  else:
    return (classes[max(set(lst), key=lst.count)])

def avg(lst):
  if len(lst) == 0:
    return -1
  else:
    return sum(lst) / len(lst)

classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
dict_list = [{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0}]
class_wrong = [[],[],[],[],[]]
snr_wrong = [[],[],[],[],[]]
with torch.no_grad():
  for i in range(len(test_data)):
    data_file = test_data[i]
    inputs, labels, snr = data_file
 
    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    if (predicted.item() != labels.item()):
      # class_wrong[labels.item()].append(predicted.item())
      snr_wrong[labels.item()].append(snr.item())
    dict_list[labels.item()][classes[predicted.item()]] += 1

# for i in range(5):
#   print('misclassification & snr of %5s : %5s, %2f' %(
#     # classes[i], most_common(class_wrong[i]), avg(snr_wrong[i])))
#     classes[i], most_common(class_wrong[i]), avg(snr_wrong[i])))

for i in range(5):
  print('avg snr wrong & classifications of %5s : %2f' %(
    # classes[i], most_common(class_wrong[i]), avg(snr_wrong[i])))
    classes[i], avg(snr_wrong[i])))
  print(dict_list[i])

print("******************************")
print()

print()
print("******************************")
print("STATISTIC FOR TRAINING DATA")
print()

correct = 0
total = 0
network = network.float()
training_data = training_data[:500]
with torch.no_grad():
  for i in range(len(training_data)):
    data_file = training_data[i]
    inputs, labels = data_file

    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 500 training images: %d %%' % (
    100 * correct / total))

# which classes performed well, which did not perform well
classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
with torch.no_grad():
  for i in range(len(training_data)):
    data_file = training_data[i]
    inputs, labels = data_file

    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).item()
    for i in range(4):
      # print(labels)
      label = labels.item()
      class_correct[label] += c
      class_total[label] += 1

for i in range(5):
  print('Accuracy of %5s : %2d %%' % (
    classes[i], 100 * class_correct[i] / class_total[i]))

classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
dict_list = [{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0},
{'llbb': 0, 'llnb': 0, 'slbb': 0, 'slnb': 0, 'noise': 0}]
class_wrong = [[],[],[],[],[]]
with torch.no_grad():
  for i in range(len(training_data)):
    data_file = training_data[i]
    inputs, labels = data_file
 
    inputs = inputs.unsqueeze(0)
    outputs = network(inputs.float())
    _, predicted = torch.max(outputs.data, 1)
    # if (predicted.item() != labels.item()):
    #   class_wrong[labels.item()].append(predicted.item())
    dict_list[labels.item()][classes[predicted.item()]] += 1

for i in range(5):
  print('classifications of %5s : ' %(classes[i]))
  print(dict_list[i])

print("******************************")
print()


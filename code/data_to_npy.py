import numpy as np
import matplotlib.pyplot as plt
data_path = 'C:/Santosh/AI/K-Comp/Ion/Data/wdwk/'
train_data_file = data_path + 'train.csv'
test_data_file = data_path + 'test.csv'

def get_data(filename, train=True):
  
    if(train):
        with open(filename) as training_file:
            split_size = 10
            data = np.loadtxt(training_file, delimiter=',', skiprows=1)
            signal = data[:,1]
            channels = data[:,2]
            signal = np.array_split(signal, split_size)
            channels = np.array_split(channels, split_size)
            data = None
        return np.array(signal), np.array(channels)
    else:
       with open(filename) as training_file:
            split_size = 4
            data = np.loadtxt(training_file, delimiter=',', skiprows=1)
            signal = data[:,1]
            signal = np.array_split(signal, split_size)
            data = None
       return np.array(signal)

train_signal , train_channels = get_data(train_data_file)
test_signal = get_data(test_data_file, train=False)

plt.plot(np.arange(5000000),train_signal.flatten(),'r')

plt.plot(np.arange(2000000),test_signal.flatten(),'r')

print(train_signal.shape)
print(train_channels.shape)
print(test_signal.shape)

np.save(data_path + 'train_signal.npy', train_signal)
np.save(data_path + 'train_channels.npy', train_channels)
np.save(data_path + 'test_signal.npy', test_signal)


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_path = 'C:/Santosh/AI/K-Comp/Ion/Data/'
train_signal_file = data_path + 'train_signal.npy'
train_channels_file = data_path + 'train_channels.npy'
test_signal_file = data_path + 'test_signal.npy'

train_signal = np.load(train_signal_file)
train_channels = np.load(train_channels_file)
test_signal = np.load(test_signal_file)

valid_signal = train_signal[:,400000:]
valid_channels = train_channels[:,400000:]
train_signal = train_signal[:,:400000]
train_channels = train_channels[:,:400000]

window_size = 10
batch_size = 50
shuffle_buffer = 1000

def windowed_dataset(all_series, all_channels, window_size, batch_size, shuffle_buffer):
    
    all_dataset = []
    cnt = 0
    for series in all_series:
        
        for i in range(10000,15000):
            all_dataset = np.concatenate([all_dataset,series[i-window_size+1:i+1]])
            all_dataset = np.concatenate([all_dataset,[all_channels[cnt][i]]])
        cnt+=1
    #print(all_dataset)
    series = np.array(all_dataset)
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=window_size+1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

tf.keras.backend.clear_session()

dataset = windowed_dataset(train_signal, train_channels, window_size, batch_size, shuffle_buffer)



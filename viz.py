import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_multi_curve(filename, title):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data = np.array([np.array(x) for x in data])
        data = data.reshape(-1, 2)
        y = np.hsplit(data, 2)[1]
        print(y)
        plt.plot(y)
        plt.title(title)
        plt.show()

def plot_single_curve(filename, title):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data = np.array(data)
        plt.plot(data)
        plt.title(title)
        plt.show()


plot_multi_curve('train_losses.pkl', "3 Layer CNN Training Loss")
plot_single_curve('test_losses.pkl', "3 Layer CNN Test Loss")
plot_single_curve('model_accuracy.pkl', "3 Layer CNN Accuracy")
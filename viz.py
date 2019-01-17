import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_multi_curve(filename, title, xl='Batch', yl='Loss'):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data = np.array([np.array(x) for x in data])
        data = data.reshape(-1, 2)
        y = np.hsplit(data, 2)[1]
        print(y)
        plt.plot(y)
        plt.title(title)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.show()

def plot_single_curve(filename, title, xl='Batch', yl='Loss'):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        data = np.array(data)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.show()


plot_multi_curve('outputs/3_layer_train_losses.pkl', "3 Layer CNN Training Loss", xl='Batch', yl='Loss')
plot_single_curve('outputs/3_layer_test_losses.pkl', "3 Layer CNN Test Loss", xl='Epoch', yl='Loss')
plot_single_curve('outputs/3_layer_accuracy.pkl', "3 Layer CNN Accuracy", xl='Epoch', yl='Accuracy')

plot_multi_curve('outputs/three_layer_train_losses.pkl', "3 Layer with Interlayer Routing CNN Training Loss", xl='Batch', yl='Loss')
plot_single_curve('outputs/three_layer_test_losses.pkl', "3 Layer with Interlayer Routing CNN Test Loss", xl='Epoch', yl='Loss')
plot_single_curve('outputs/three_layer_model_accuracy.pkl', "3 Layer with Interlayer Routing CNN Accuracy", xl='Epoch', yl='Accuracy')


plot_multi_curve('outputs/multi_output_train_losses.pkl', "Early Output 3 Layer CNN Training Loss", xl='Batch', yl='Loss')
plot_single_curve('outputs/multi_output_test_losses.pkl', "Early Output 3 Layer CNN Test Loss", xl='Epoch', yl='Loss')
plot_single_curve('outputs/multi_output_model_accuracy.pkl', "Early Output 3 Layer CNN Accuracy", xl='Epoch', yl='Accuracy')
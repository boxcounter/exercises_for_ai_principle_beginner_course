import dataset
import numpy as np
from tqdm import tqdm
import plot_utils

dataset_size = 100
xs, ys = dataset.get_beans(dataset_size)

w1 = np.random.random()
w2 = np.random.random()
b = np.random.random()
alpha = 0.1
epochs = 500

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x1s, x2s):
    zs = w1 * x1s + w2 * x2s + b
    return sigmoid(zs)

for _ in tqdm(range(epochs)):
    for i in range(dataset_size):
        x1 = xs[i, 0]
        x2 = xs[i, 1]
        y = ys[i]

        # ForwardProp
        z = w1 * x1 + w2 * x2 + b
        a = sigmoid(z)

        e = (a - y) ** 2

        # BackProp
        de_da = 2 * (a - y)
        da_dz = a * (1 - a) # deravative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        dz_dw1 = x1
        dz_dw2 = x2
        dz_db = 1

        gradient_w1 = de_da * da_dz * dz_dw1
        gradient_w2 = de_da * da_dz * dz_dw2
        gradient_db = de_da * da_dz * dz_db

        w1 = w1 - alpha * gradient_w1
        w2 = w2 - alpha * gradient_w2
        b = b - alpha * gradient_db

print(f"w1 = {w1}, w2 = {w2}, b = {b}\n")
print("training complete")
plot_utils.show_scatter_surface(xs, ys, predict)
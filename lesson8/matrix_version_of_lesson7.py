import dataset
import numpy as np
import plot_utils
from tqdm import tqdm

dataset_size = 100
X, Y = dataset.get_beans(dataset_size)

plot_utils.show_scatter(X, Y)

W = np.array([np.random.random(), np.random.random()])
B = np.array([np.random.random()])
alpha = 0.05
epochs = 500

def sigmoid(param_X):
    return 1 / (1 + np.exp(-param_X))

def predict(param_X):
    Z = param_X.dot(W.T) + B
    return sigmoid(Z)

for _ in tqdm(range(epochs)):
    for i in range(dataset_size):
        Xi = X[i] # 1 row and 2 colomns (abbr: 1*2)
        Yi = Y[i] # 1 * 1

        # ForwardProp
        Z = Xi.dot(W.T) + B # 1*1 = 1*2 . 2*1 + 1
        A = sigmoid(Z) # 1*1

        E = (A - Yi) ** 2 # 1*1

        # BackProp
        dE_dA = 2 * (A - Yi) # 1*1
        dA_dZ = A * (1 - A) # 1*1, deravative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        dZ_dW = Xi # 1*2
        dZ_dB = 1

        gradient_W = dE_dA * dA_dZ * dZ_dW # The Chain Rule
        gradient_B = dE_dA * dA_dZ * dZ_dB

        W = W - alpha * gradient_W
        B = B - alpha * gradient_B

print(f"W = {W}, B = {B}\n")
print(f"trainning complete")

plot_utils.show_scatter_surface(X, Y, predict)
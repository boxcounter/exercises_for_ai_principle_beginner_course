# lesson vedio: https://www.bilibili.com/cheese/play/ep7307
import dataset
import matplotlib.pyplot as plot
import numpy as np
from tqdm import tqdm

dataset_size = 100
xs, ys = dataset.get_beans(dataset_size)
epochs = 1000
alpha = 0.1

#
# w12_3: from 1st unit to 2nd unit of 3rd hidden layer
#

b1_1 = np.random.random()
b2_1 = np.random.random()
w11_1 = np.random.random()
w12_1 = np.random.random()

b1_2 = np.random.random()
w11_2 = np.random.random()
w21_2 = np.random.random()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict():
    zs1_1 = w11_1 * xs + b1_1
    as1_1 = sigmoid(zs1_1)

    zs2_1 = w12_1 * xs + b2_1
    as2_1 = sigmoid(zs2_1)

    zs1_2 = w11_2 * as1_1 + w21_2 * as2_1 + b1_2
    as1_2 = sigmoid(zs1_2)
    return as1_2

for ep in tqdm(range(epochs)):
    for i in range(dataset_size):
        x = xs[i]
        y = ys[i]

        #
        # ForewardProp
        # 

        z1_1 = w11_1 * x + b1_1
        a1_1 = sigmoid(z1_1)

        z2_1 = w12_1 * x + b2_1
        a2_1 = sigmoid(z2_1)

        z1_2 = w11_2 * a1_1 + w21_2 * a2_1 + b1_2
        a1_2 = sigmoid(z1_2)

        e = (a1_2 - y) ** 2

        #
        # BackProp for layer 2
        #

        de_da1_2 = 2 * (a1_2 - y) # e = a1_2^2 - 2*a1_2*y + y^2, deravative = 2*a1_2 - y
        da1_2_dz1_2 = a1_2 * (1 - a1_2) # deravative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        
        dz1_2_dw11_2 = a1_1
        gradient_w11_2 = de_da1_2 * da1_2_dz1_2 * dz1_2_dw11_2

        dz1_2_dw21_2 = a2_1
        gradient_w12_2 = de_da1_2 * da1_2_dz1_2 * dz1_2_dw21_2

        dz1_2_db1_2 = 1
        gradient_b1_2 = de_da1_2 * da1_2_dz1_2 * dz1_2_db1_2

        w11_2 = w11_2 - alpha * gradient_w11_2
        w21_2 = w21_2 - alpha * gradient_w12_2
        b1_2 = b1_2 - alpha * gradient_b1_2

        #
        # BackProp for layer 1
        #

        # unit 1
        dz1_2_da1_1 = w11_2
        da1_1_dz1_1 = a1_1 * (1 - a1_1)

        dz1_1_dw11_1 = x
        gradient_w11_1 = de_da1_2 * da1_2_dz1_2 * dz1_2_da1_1 * da1_1_dz1_1 * dz1_1_dw11_1

        dz1_1_b1_1 = 1
        gradient_b1_1 = de_da1_2 * da1_2_dz1_2 * dz1_2_da1_1 * da1_1_dz1_1 * dz1_1_b1_1

        w11_1 = w11_1 - alpha * gradient_w11_1
        b1_1 = b1_1 - alpha * gradient_b1_1

        # unit 2
        dz1_2_da2_1 = w21_2
        da2_1_dz2_1 = a2_1 * (1 - a2_1)

        dz2_1_dw12_1 = x
        gradient_w12_1 = de_da1_2 * da1_2_dz1_2 * dz1_2_da2_1 * da2_1_dz2_1 * dz2_1_dw12_1

        dz2_1_db2_1 = 1
        gradient_b2_1 = de_da1_2 * da1_2_dz1_2 * dz1_2_da2_1 * da2_1_dz2_1 * dz2_1_db2_1

        w12_1 = w12_1 - alpha * gradient_w12_1
        b2_1 = b2_1 - alpha * gradient_b2_1

    if epochs % 50 == 0:
        ys_predict = predict()
        
        plot.clf()
        plot.scatter(xs, ys)
        plot.plot(xs, ys_predict)
        plot.pause(0.01)

ys_predict = predict()
plot.clf()
plot.scatter(xs, ys)
plot.plot(xs, ys_predict)
plot.show()
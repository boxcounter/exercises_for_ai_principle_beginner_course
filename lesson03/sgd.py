# stochastic gredient descent

import dataset
import matplotlib.pyplot as plot
import numpy

dataset_size = 100
xs, ys = dataset.get_beans(dataset_size)

w = 0.1
alpha = 0.1

training_round = 100

for i in range(dataset_size):
    x = xs[i]
    y = ys[i]
    for _ in range(training_round):
        y_predict = w * x
        # e = (y_predict - y) ** 2
        # e = (w * x - y) **2
        # e = w^2 * x^2 - 2wxy + y^2
        # e = x^2 * w^2 - 2xy * w + y^2
        # k = e' = 2(x^2)*w - 2xy
        k = 2 * (x ** 2) * w - 2 * x * y
        w = w - alpha * k

    ys_predict = w * xs

    plot.clf()
    plot.xlim(0, 1)
    plot.ylim(0, 1.4)
    plot.scatter(xs, ys)
    plot.plot(xs, ys_predict)
    plot.pause(0.01)

plot.show()

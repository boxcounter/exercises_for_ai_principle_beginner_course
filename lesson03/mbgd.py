# mini-batch gredient descent

import dataset
import matplotlib.pyplot as plot
import numpy

batch_size = 20
batch_count = 5
xs, ys = dataset.get_beans(batch_count * batch_size)

w = 0.1
alpha = 0.1
training_round = 100

for r in range(batch_count):
    start = r * batch_size
    stop = start + batch_size - 1
    print(f"{start} -> {stop}\n")

    xb = xs[start : stop] # b means batch
    yb = ys[start : stop]

    for _ in range(training_round):
        yb_predict = w * xb
        # e = (yb_predict - y) ^ 2
        # e = yb_predict^2 - 2*yb_predict*y + y^2
        # e = (w * xb)^2 - 2 * w * xb * y + y^2
        #   = (xb^2 * w^2) - (2 * xb * y * w) + y^2
        # k（slope） = e' = de/dw = 2 * xb^2 * w - 2 * xb * y
        k = 2 * (xb ** 2) * w - 2 * xb * yb
        w = w - alpha * numpy.sum(k) / batch_size

        yb_predict = w * xb
        plot.clf()
        plot.xlim(0, 1)
        plot.ylim(0, 1.4)
        plot.scatter(xb, yb)
        plot.plot(xb, yb_predict)
        plot.pause(0.01)

ys_predict = w * xs
plot.scatter(xs, ys)
plot.plot(xs, ys_predict)
plot.show()
import dataset
import matplotlib.pyplot as plot
import numpy

# z = wx + b
# y_predict = sigmoid(z)                # activation function
# e = (y_predict - y) ^ 2               # cost function
#
# the target is de/dw and de/db, use The Chain Rule:
#   de/dw = de/dy_predict * dy_predict/dz * dz/dw
#   de/db = de/dy_predict * dy_predict/dz * dz/db
#
#   e = y_predict^2 - 2(y_predict*y)+ y^2
#
#   de/dy_predict = 2y_predict - 2y + 0
#                 = 2(y_predict - y)
#
#   dy_predict/dz = 1/(1+e^-z) * (1-1/(1+e^-z))  # derivative of the sigmoid
#   dz/dw = x
#   dz/db = 1

batch_size = 20
batch_count = 5
xs, ys = dataset.get_beans(batch_size * batch_count)

w = 0.1
b = 0.1
alpha = 2

epochs = 5000

for r in range(epochs):
    for i in range(batch_count * batch_size):
        start = i * batch_size
        stop = start + batch_size - 1
        # print(f"{start} -> {stop}\n")

        xb = xs[start : stop]
        yb = ys[start : stop]

        zb = w * xb + b
        yb_predict = 1 / (1 + numpy.exp(-1 * zb))

        dedy_predict = 2 * (yb_predict - yb)
        dy_predictdz = yb_predict * (1 - yb_predict)
        dzdw = xb
        dzdb = 1

        dedw = numpy.sum(dedy_predict * dy_predictdz * dzdw) / batch_size
        dedb = numpy.sum(dedy_predict * dy_predictdz * dzdb) / batch_size

        w = w - alpha * dedw
        b = b - alpha * dedb

    # print(f"w = {w}, b = {b}\n")
    zs_predict = w * xs + b
    ys_predict = 1 / (1 + numpy.exp(-1 * zs_predict))

    if r % 100 == 0:
        plot.clf()
        plot.scatter(xs, ys)
        plot.plot(xs, ys_predict)
        plot.pause(0.01)

zs_predict = w * xs + b
ys_predict = 1 / (1 + numpy.exp(-1 * zs_predict))

plot.clf()
plot.scatter(xs, ys)
plot.plot(xs, ys_predict)
plot.show()
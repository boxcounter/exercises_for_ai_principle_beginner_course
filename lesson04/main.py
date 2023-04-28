import dataset
import matplotlib.pyplot as plot
import numpy

# predict function: y_predict = wx + b
# cost function: 
#    e = (y_predict - y)^2
#      = (wx + b - y)^2
#      = (w^2 * x^2) + b^2 + y^2 + (2 * w * x * b) + (2 * wx * (-y)) + (2 * b * (-y))
#      = (x^2 * w^2) + b^2 + (2 * x * w * b) - (2 * x * y * w) - (2 * y * b) + y^2
# gredient (slope for w and b)
#    e'(w) = de/dw 
#          = (2 * x^2 * w) + 0 + (2 * x * b) - (2 * x * y) - 0 + 0
#          = (2 * x^2 * w) + 2 * x * (b - y)
#          = 2 * (x^2 * w + x * (b-y))
#    e'(b) = de/db
#          = 0 + (2 * b) + (2 * x * w) - 0 - (2 * y) + 0
#          = 2 * b + 2 * x * w - 2 * y
#          = 2 * (b + x * w - y)

# parameters for mini-batch stochastic gredient descent
batch_size = 20
batch_count = 5
xs, ys = dataset.get_beans(batch_size * batch_count)

w = 0.1
b = 0.1
alpha = 0.07

training_round = 2000

for i in range(batch_count):
    start = i * batch_size
    stop = start + batch_size - 1
    print(f"{start} -> {stop}\n")

    xb = xs[start : stop]
    yb = ys[start : stop]

    for _ in range(training_round):
        dw = numpy.sum(2 * (xb ** 2 * w + xb * (b - yb))) / batch_size # d means derivative
        db = numpy.sum(2 * (b + xb * w - yb)) / batch_size
        w = w - alpha * dw
        b = b - alpha * db

    print(f"w = {w}, b = {b}\n")
    ys_predict = w * xs + b
    # plot figure for current batch dataset and predict function
    plot.clf()
    plot.xlim(0, 1.2)
    plot.ylim(0, 1.4)
    plot.scatter(xs, ys)
    plot.plot(xs, ys_predict)
    plot.pause(0.01)


print(f"w = {w}, b = {b}\n")
print("done.")
import dataset
import matplotlib.pyplot as plot
import numpy

dataset_size = 100
xs, ys = dataset.get_beans(dataset_size)

w = 0.1
alpha = 0.1
training_round = 100

for _ in range(training_round):
    ys_predict = w * xs
    es = (ys_predict - ys)**2
    e = numpy.sum(es) * alpha
    w += e / dataset_size

print(f"w = {w}\n")
ys_predict = w * xs

plot.scatter(xs, ys)
plot.plot(xs, ys_predict)
plot.show()


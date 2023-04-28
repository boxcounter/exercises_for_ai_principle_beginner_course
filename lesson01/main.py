import dataset
import matplotlib.pyplot as plot
import numpy

count = 100
xs, ys = dataset.get_beans(count)

plot.title("Size-Toxicity Function", fontsize = 12)
plot.xlabel("Bean Size")
plot.ylabel("Toxicity")
plot.scatter(xs, ys)
# plot.show()

alpha = 0.1
w = 0

for i in range(count):
    x_sample = xs[i]
    y_sample = ys[i]

    y = w * x_sample
    e = y_sample - y
    w += e * x_sample * alpha

y_predict = xs * w
plot.plot(xs, y_predict)
plot.show()


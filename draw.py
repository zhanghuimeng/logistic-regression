import pickle
import numpy as np
from matplotlib import pyplot as plt

lr_list = [0.1, 0.05, 0.01, 0.001]
color_list = ["b", "g", "r", "k"]
d_list = []
for lr in lr_list:
    with open("output/gd_lr-%f.pickle" % lr, "rb") as f:
        d_list.append(pickle.load(f))

for i in range(len(lr_list)):
    print("lr=%f, acc=%f" % (lr_list[i], d_list[i]["acc"][-1]))

plt.subplot(1, 2, 1)
for i in range(len(lr_list)):
    plt.plot(d_list[i]["acc"], color=color_list[i], label="lr=%f" % lr_list[i])
plt.xlabel("step")
plt.ylabel("test accuracy")
plt.ylim((0.99, 1.00))
plt.legend()

plt.subplot(1, 2, 2)
for i in range(len(lr_list)):
    plt.plot(d_list[i]["ll"], color=color_list[i], label="lr=%f" % lr_list[i])
plt.xlabel("step")
plt.ylabel("log-likelihood")
plt.legend()

plt.show()

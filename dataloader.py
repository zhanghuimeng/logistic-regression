import os
import numpy as np


def load(filename="a9a", d=123):
    with open(filename, "r") as f:
        lines = f.readlines()
    N = len(lines)
    feature = np.zeros([N, d])
    label = np.zeros(N)
    for i, line in enumerate(lines):
        points = line.strip().split(" ")
        if points[0] == "+1":
            label[i] = 1
        else:
            label[i] = 0
        for point in points:
            f = int(point.split(":")[0])
            feature[i][f - 1] = 1

    print("Read %d data" % N)
    return feature, label


# Test
feature, label = load("a9a/a9a")
print(feature)
print(label)
feature, label = load("a9a/a9a.t")
print(feature)
print(label)

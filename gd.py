import argparse
import os
import pickle
import numpy as np
from dataloader import load


def gradient(x, y, w):
    # Calculate the gradient
    N, D = x.shape
    g = np.zeros(D)
    for i in range(N):
        l = np.exp(-np.dot(w, x[i]))
        mu = 1 / (1 + l)
        g += x[i] * (y[i] - mu)
    return g


def log_likelihood(x, y, w, lamb=0.0):
    ll = 0.0
    N = x.shape[0]
    for i in range(N):
        l = 1 + np.exp(np.dot(w, x[i]))
        ll += y[i] * np.dot(w, x[i]) - np.log(l)
    ll -= 0.5 * lamb * np.linalg.norm(w)**2
    return ll


def evaluate(x, y, w):
    N = x.shape[0]
    prediction = []
    for i in range(N):
        l = np.exp(-np.dot(w, x[i]))
        p = 1 / (1 + l)
        if p > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)
    return np.array(prediction)


def gd(train, test, lr, patience):
    D = train["feature"].shape[1]
    # w = np.random.random(D)
    w = np.random.uniform(0, 0.1, D)
    acc_list = []
    ll_list = []
    best = None
    wait = 0
    step = 0
    while wait < patience:
        g = gradient(train["feature"], train["label"], w)
        # print("g: %f" % np.mean(g))
        w += lr * g
        # print("w: %f" % np.mean(w))
        prediction = evaluate(test["feature"], test["label"], w)
        acc = np.mean(prediction == test["label"])
        ll = log_likelihood(train["feature"], train["label"], w)
        if best is None or acc > best:
            best = acc
            wait = 0
        else:
            wait += 1
        print("step=%d, acc=%f, ll=%f" % (step, acc, ll))
        acc_list.append(acc)
        ll_list.append(ll)
        step += 1
    
    return acc_list, ll_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using gradient descent to solve logistic regression.")
    parser.add_argument("--data", type=str, default="a9a",
                        help="Directory of the a9a data.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience to stop.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output log-likelihood and accuracy during training to pickle file.")
    args = parser.parse_args()

    train = {}
    test = {}
    train["feature"], train["label"] = load(os.path.join(args.data, "a9a"))
    test["feature"], test["label"] = load(os.path.join(args.data, "a9a.t"))
    # add for w_0
    ones = np.ones([train["feature"].shape[0], 1])
    train["feature"] = np.append(train["feature"], ones, axis=1)
    ones = np.ones([test["feature"].shape[0], 1])
    test["feature"] = np.append(test["feature"], ones, axis=1)

    acc_list, ll_list = gd(train, test, args.lr, args.patience)

    if args.output is None:
        with open("output/gd_lr-%f.pickle" % args.lr, "wb") as f:
            d = {"acc": acc_list, "ll": ll_list}
            pickle.dump(d, f)

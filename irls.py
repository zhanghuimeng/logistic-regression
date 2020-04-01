import argparse
import os
import numpy as np
from sklearn.model_selection import KFold
from dataloader import load
from gd import evaluate, log_likelihood


def update(x, y, w, lamb):
    # calculate gradient/H
    N, D = x.shape
    X = x.T
    r = []
    mu = []
    for i in range(N):
        l = np.exp(-np.dot(w, x[i]))
        mu_i = 1 / (1 + l)
        mu.append(mu_i)
        r.append(mu_i * (1 - mu_i))
    mu = np.array(mu)
    R = np.diag(r)
    z = y - mu
    G = X.dot(z.reshape(N, 1)) - lamb * w.reshape(D, 1)
    H = -X @ R @ X.T - lamb * np.eye(D)
    # print("X mean = %f" % np.mean(X))
    # print("(y-mu) mean = %f" % np.mean(z))
    # print("G mean = %f" % np.mean(G))
    # print("H mean = %f" % np.mean(H))
    # print("inv(H) mean = %f" % np.mean(np.linalg.pinv(H)))
    return np.linalg.pinv(H) @ G


def irls(train, test, lamb, patience):
    D = train["feature"].shape[1]
    # w = np.random.random(D)
    w = np.random.uniform(0, 0.1, D)
    acc_list = []
    ll_list = []
    best = None
    wait = 0
    step = 0

    while wait < patience:
        upd = update(train["feature"], train["label"], w, lamb)
        upd = upd.ravel()
        w = w - upd
        prediction = evaluate(test["feature"], test["label"], w)
        acc = np.mean(prediction == test["label"])
        if best is None or acc > best:
            best = acc
            wait = 0
        else:
            wait += 1
        ll = log_likelihood(train["feature"], train["label"], w, lamb)
        print("step=%d, acc=%f, ll=%f" % (step, acc, ll))
        acc_list.append(acc)
        ll_list.append(ll)
        step += 1
    
    return w, acc_list, ll_list


parser = argparse.ArgumentParser(description="Using gradient descent to solve logistic regression.")
parser.add_argument("--data", type=str, default="a9a",
                    help="Directory of the a9a data.")
parser.add_argument("--lamb", type=float, default=0.0,
                    help="Regularization constant.")
parser.add_argument("--cross", action="store_true",
                    help="Do cross validation or not.")
parser.add_argument("--patience", type=int, default=5,
                    help="Patience to stop.")
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

if not args.cross:
    w, acc_list, ll_list = irls(train, test, args.lamb, args.patience)
    print("Final step: %d" % len(acc_list))
    print("Final training acc: %f" % evaluate(train["feature"], train["label"], w))
    print("Final testing acc: %f" % acc_list[-1])
    print("Final L2 norm of w: %f" % np.linalg.norm(w)**2)
else:
    kf = KFold(n_splits=10, shuffle=True)
    step_list = []
    train_acc_list = []
    val_acc_list = []
    w_list = []

    fold = 0
    for train_index, val_index in kf.split(X):
        x_train, x_val = train["feature"][train_index], train["feature"][val_index]
        y_train, y_val = train["label"][train_index], train["label"][val_index]

        

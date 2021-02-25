import numpy as np
import random

def kmean(X, k=9, epoch=20):
    """
    K=9 for Yolo
    K=6 for Yolo_tiny
    """
    X = np.array(X)
    C = X[random.sample(range(len(X)), k), :]
    Xt = np.tile(X, (k, 1, 1))
    for i in range(epoch):
        C = np.expand_dims(C, 1)
        Ct = np.tile(C, (1, len(X), 1))
        c = np.argmin(np.sum(pow(Xt - Ct, 2), axis=2), axis=0)
        C = [X[c == k].mean(axis = 0) for k in range(k)]
    return np.array(C) , c

def main():
    X=[[1,0], [0,1], [-1,0], [100,101], [100,100], [50,101]]
    k=2
    C, c = kmean(X, k, 10)
    print(C)
    print(c)


if __name__=='__main__':
    main()
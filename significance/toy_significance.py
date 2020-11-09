import numpy as np

import argparse
import sys
import os

from utils import generate_data, unison_shuffled_copies

from lassonet.lassonet_trainer import lassonet_trainer

import uuid
import datetime
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--p', type=int, default=50)
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--n_b', type=int, default=1)
    parser.add_argument('--noise_var', type=float, default=.6)
    parser.add_argument('--weakness', type=float, default=.8)
    parser.add_argument('--perm', type=str, choices=['original', 'permuted'], default='original')
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    n = args.n
    p = args.p
    n_bootstraps = args.n_bootstraps
    n_b = args.n_b
    noise_var = args.noise_var
    weakness = args.weakness

    test_size = 10

    X, y, b = generate_data(n + test_size, p, n_b=n_b, noise_var=noise_var, seed=123)
    y = np.expand_dims(y, axis=-1)
    X_test = X[n:n + test_size]
    y_test = y[n:n + test_size]
    X_train = X[:n]
    y_train = y[:n]

    X_train_perm = X_train.copy()
    y_train_perm = y_train.copy()
    if args.perm == 'permuted':
        np.random.shuffle(y_train_perm)
    m = int(np.floor(n / 2.))

    figure_dir = str(datetime.datetime.now()).replace(' ', ',') + '_' + str(uuid.uuid4())
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    for _ in range(n_bootstraps):
        start = time.time()
        X_train_perm, y_train_perm = unison_shuffled_copies(X_train_perm, y_train_perm)
        weights = 1. - (1. - weakness) * np.random.randint(2, size=p)

        lassonet_trainer('toy_' + args.perm + '_' + str(uuid.uuid4()),
                         (weights * X_train_perm[:m], y_train_perm[:m]),
                         (X_test, y_test),
                         utils={'figure_dir': figure_dir,
                                'criterion': 'MSE'})
        end = time.time()
        print('elapsed time:', end - start)

if __name__ == '__main__':
    main()

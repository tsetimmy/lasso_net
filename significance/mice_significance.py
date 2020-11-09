import numpy as np
import pickle

import argparse
import sys
import os

from utils import unison_shuffled_copies


import torch
from lassonet.lassonet_trainer import lassonet_trainer

import uuid
import datetime


import time
def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--name', type=str, default='name')
    #parser.add_argument('--figure_dir', type=str, default='')
    parser.add_argument('--n_bootstraps', type=int, default=100)
    parser.add_argument('--weakness', type=float, default=.8)
    parser.add_argument('--perm', type=str, choices=['original', 'permuted'], default='original')
    args = parser.parse_args()

    print(sys.argv)
    print(args)

    train, test = pickle.load(open('../pickle_files/mice_data.pickle', 'rb'))
    X_train, y_train = train

    n_bootstraps = args.n_bootstraps
    weakness = args.weakness
    n, p = X_train.shape
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

        lassonet_trainer(args.perm + '_' + str(uuid.uuid4()),
                         (weights * X_train_perm[:m], y_train_perm[:m]),
                         test,
                         utils={'figure_dir': figure_dir})
        end = time.time()
        print('elapsed time:', end - start)

if __name__ == '__main__':
    main()

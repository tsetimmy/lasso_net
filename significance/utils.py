import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def generate_data(n, p, n_b=1, noise_var=1., seed=None):
    if seed is not None:
        random_state = np.random.get_state()
        np.random.seed(seed)

    X = np.random.normal(size=[n, p])
    b = np.zeros(p)
    b[:n_b] = 1.
    np.random.shuffle(b)

    e = np.random.normal(scale=noise_var**.5, size=n)

    y = X @ b + e

    if seed is not None:
        np.random.set_state(random_state)

    return X, y, b

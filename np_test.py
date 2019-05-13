import numpy as np


class NpTest:
    def __init__(self):
        pass

    def test(self):
        x = np.array([1, 2, 3, 4])
        print(np.mean(x), np.std(x))
        x_bn = (x - np.mean(x)) / np.std(x)
        print(x_bn)


if __name__ == '__main__':
    n = NpTest()
    n.test()

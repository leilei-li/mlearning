import numpy as np


class NpTest:
    def __init__(self):
        pass

    def test(self):
        A = [1, 1, 1, -1]
        A = np.array(A)
        A = A.reshape([2, 2])
        A_T = A.T
        b = np.array([0, 2]).reshape([-1,1])
        print(np.linalg.inv(np.dot(A_T, A)).dot(A_T).dot(b))


if __name__ == '__main__':
    n = NpTest()
    n.test()

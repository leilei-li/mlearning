import numpy as np
import matplotlib.pyplot as plt


class Test:
    def __init__(self):
        self.x = np.arange(-5, 5, 1e-2)
        pass

    def __draw_pic(self, x, y):
        plt.figure(1)
        plt.plot(x, y)
        plt.show()

    def draw_a_picture(self, func_name, arg_dic=None):
        if func_name == "gaussian_distribution":
            mu = arg_dic['mu']
            sigma = arg_dic['sigma']
            self.draw_gaussian_distribution(mu=mu, sigma=sigma)
        if func_name == "tanh":
            self.draw_tanh()
        if func_name == "cross_entropy":
            self.draw_cross_entropy()

    def draw_gaussian_distribution(self, mu, sigma):
        x = self.x
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.power(np.e, -((x - mu) ** 2) / (2 * sigma ** 2))
        self.__draw_pic(x, y)

    def draw_tanh(self):
        x = self.x
        y = np.tanh(x)
        self.__draw_pic(x, y)

    def draw_cross_entropy(self):
        x = self.x
        y = -(x * np.log2(x) + (1 - x) * np.log2(1 - x))
        self.__draw_pic(x, y)


if __name__ == '__main__':
    t = Test()
    t.draw_a_picture(func_name='cross_entropy')

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
        if func_name == "derivatives":
            self.draw_derivatives()

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

    def draw_derivatives(self):
        """
        通过导数公式求导
        y=sin(x)
        :return:
        """
        x = self.x
        y = np.cos(x)
        h = 1e-6
        d_y = (np.sin(x + h) - np.sin(x - h)) / (2 * h)
        plt.plot(x, y + 0.5)
        plt.plot(x, d_y)
        plt.show()

    def draw_three_d(self):
        xx = np.arange(-np.pi, np.pi + 0.01, 0.01)
        yy = np.arange(-np.pi, np.pi + 0.01, 0.01)
        x, y = np.meshgrid(xx, yy)
        z = np.sin(x ** 2) + np.sin(y ** 2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
        ax.plot_surface(x, y, z)
        plt.show()

    def draw_tpp_zhifang_errorbar(self):
        y = [0.2476, 0.0882, 0.0788]
        y_2 = [0.25, 0.0883, 0.0883]
        error_bar = [0.0152, 0.0103, 0.01122]
        x = np.arange(len(y))
        plt.figure(1)
        err_attr = {"elinewidth": 2, "ecolor": "black", "capsize": 3}
        x_tick_label = ('$A_1$', '$A_2$', '$A_3$')
        x_2_tick_label = ('$B_1$', '$B_2$', '$B_3$')
        width = 0.25
        b1 = plt.bar(x, y, width=width, yerr=error_bar, error_kw=err_attr, tick_label=x_tick_label)
        b2 = plt.bar(x + width + 0.02, y_2, width=width, tick_label=x_tick_label)
        plt.legend([b1, b2], ['$y_1$ with error bar', '$y_2$'])
        plt.ylabel('$y=\sum_i x_i$', fontsize=16)
        plt.xlabel('$x_i$', fontsize=16)
        plt.grid(axis='y', color='grey', linewidth='0.5')
        # plt.savefig('6-9.eps', dpi=300)
        plt.show()


if __name__ == '__main__':
    t = Test()
    # t.draw_a_picture(func_name='derivatives')
    t.draw_tpp_zhifang_errorbar()

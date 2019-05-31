from tensorflow.python import keras
import numpy as np
from reappearance.record import LossHistory
import matplotlib.pyplot as plt
import random


class NetWork:
    def __init__(self):
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.init_mnist()

    def init_mnist(self):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data(
            '/Users/lileilei/Documents/IdeaProject/mlearning/mnist/keras_mnist/mnist.npz')
        train_labels = self.label_to_onehot(train_labels)
        test_labels = self.label_to_onehot(test_labels)
        # 手动归一化
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        # 手动打乱train数据集
        random_index = [i for i in range(len(train_images))]
        random.shuffle(random_index)
        train_images = train_images[random_index]
        train_labels = train_labels[random_index]
        # mnist数据转换为四维
        # (image_num,height,weight,channel)
        train_images = np.expand_dims(train_images, axis=3)
        test_images = np.expand_dims(test_images, axis=3)
        return train_images, train_labels, test_images, test_labels

    def label_to_onehot(self, labels):
        """
        将label转化为onehot
        :param labels:
        :return:
        """
        onehot_labels = np.zeros(shape=[len(labels), 10])
        for i in range(len(labels)):
            index = labels[i]
            onehot_labels[i][index] = 1
        return onehot_labels

    def init_bp_network(self, input_shape):
        """
        初始化bp神经网络
        :return:
        """
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation=keras.activations.relu))
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Dense(units=10, activation=keras.activations.softmax))
        return model

    def init_cnn_network(self, input_shape):
        """
        初始化cnn神经网络
        :param input_shape:
        :return:
        """
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(input_shape=input_shape, filters=32, kernel_size=5, padding='same',
                                      activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                      activation=keras.activations.relu))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation=keras.activations.relu, name='full_connection_layer'))
        model.add(keras.layers.Dense(units=10, activation=keras.activations.softmax, name='output_layer'))
        return model

    def init_lstm_network(self, n_step, n_input_per_step):
        """
        初始化LSTM神经网络
        :param input_shape:
        :return:
        """
        # lstm接受的数据格式为((image_num=60000,n_step=28,n_input_per_step=28))，需要消掉channel的维度
        self.test_images = np.squeeze(self.test_images, axis=3)
        self.train_images = np.squeeze(self.train_images, axis=3)
        model = keras.Sequential()
        model.add(keras.layers.LSTM(units=128, batch_input_shape=(None, n_step, n_input_per_step)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=10, activation=keras.activations.softmax))
        return model

    def train_model(self):
        # model = self.init_bp_network(input_shape=(28, 28, 1))
        # model = self.init_cnn_network(input_shape=(28, 28, 1))
        model = self.init_lstm_network(n_step=28, n_input_per_step=28)
        history_record = LossHistory(model, self.test_images, self.test_labels)
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer,
                      loss=keras.losses.categorical_crossentropy,
                      metrics=[keras.metrics.mse,
                               keras.metrics.categorical_accuracy
                               ])
        model.fit(x=self.train_images,
                  y=self.train_labels,
                  epochs=2,
                  batch_size=32,
                  validation_split=0.2,
                  callbacks=[history_record],
                  )


def draw_categorical_accuracy():
    f = open('train_data.txt')
    history_record = eval(f.readline())
    f.close()
    x = np.array([i for i in range(len(history_record['val_acc_per_epoch']))])
    y = np.array(history_record['val_acc_per_epoch'])
    y2 = np.array(history_record['test_acc_per_epoch'])
    y3 = np.array(history_record['train_acc_per_epoch'])
    l1, = plt.plot(x, y)
    l2, = plt.plot(x, y2, linestyle='--')
    l3, = plt.plot(x, y3)
    plt.legend([l1, l2, l3], ['val_acc', 'test_acc', 'train_acc'])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    n = NetWork()
    n.train_model()
    draw_categorical_accuracy()

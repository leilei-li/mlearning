import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.batch = []

    def on_batch_end(self, batch, logs={}):
        self.batch.append(logs.get('batch'))
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('categorical_accuracy'))


def get_train_val(mnist_path):
    # mnist下载地址：https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data(mnist_path)
    print("train_images nums:{}".format(len(train_images)))
    print("test_images nums:{}".format(len(test_images)))
    return train_images, train_labels, test_images, test_labels


def one_hot(labels):
    onehot_labels = np.zeros(shape=[len(labels), 10])
    for i in range(len(labels)):
        index = labels[i]
        onehot_labels[i][index] = 1
    return onehot_labels


def mnist_net(input_shape):
    '''
    构建一个简单的全连接层网络模型：
    输入层为28x28=784个输入节点
    隐藏层120个节点
    输出层10个节点
    :param input_shape: 指定输入维度
    :return:
    '''

    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))  # 输入层
    model.add(keras.layers.BatchNormalization())  # BN层
    model.add(keras.layers.Dense(units=120, activation=tf.nn.relu))  # 隐含层
    model.add(keras.layers.core.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=10, activation=tf.nn.softmax))  # 输出层
    return model


def trian_model(train_images, train_labels, test_images, test_labels):
    #  re-scale to 0~1.0之间
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # mnist数据转换为四维
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    print("train_images :{}".format(train_images.shape))
    print("test_images :{}".format(test_images.shape))

    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    # 建立模型
    # model = mnist_net(input_shape=(28,28))
    model = mnist_net(input_shape=(28, 28, 1))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=[keras.metrics.mse,
                           keras.metrics.categorical_accuracy])
    history = LossHistory()
    model.fit(x=train_images, y=train_labels, epochs=5, batch_size=50, callbacks=[history])
    print(history)

    test_result = model.evaluate(x=test_images, y=test_labels)
    print(test_result)

    #  开始预测
    cnt = 0
    predictions = model.predict(test_images)
    for i in range(len(test_images)):
        target = np.argmax(predictions[i])
        label = np.argmax(test_labels[i])
        if target == label:
            cnt += 1
    print("correct prediction of total : %.2f" % (cnt / len(test_images)))

    # model.save('mnist-model.h5')

    return history


def draw_picture(history):
    x = np.array([i for i in range(len(history.batch))])
    y = np.array(history.losses)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    mnist_path = '/Users/lileilei/Documents/IdeaProject/mlearning/mnist/keras_mnist/mnist.npz'
    train_images, train_labels, test_images, test_labels = get_train_val(mnist_path)
    # show_mnist(train_images, train_labels)
    history = trian_model(train_images, train_labels, test_images, test_labels)
    draw_picture(history)

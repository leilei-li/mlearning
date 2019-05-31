from tensorflow.python import keras
import numpy as np


class LossHistory(keras.callbacks.Callback):
    def __init__(self, model, test_images, test_labels):
        self.model = model
        self.test_images = test_images
        self.test_labels = test_labels

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.batch = []
        self.val_acc_per_epoch = []
        self.test_acc_per_epoch = []
        self.train_acc_per_epoch = []

    def on_batch_end(self, batch, logs={}):
        self.batch.append(logs.get('batch'))
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('categorical_accuracy'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_acc_per_epoch.append(logs.get('val_categorical_accuracy'))
        self.test_acc_per_epoch.append(self.cal_accuracy_in_test_set())
        self.train_acc_per_epoch.append(logs.get('categorical_accuracy'))

    def on_train_end(self, logs=None):
        w = open('train_data.txt', 'w')
        d = {}
        d['losses'] = self.losses
        d['accuracy'] = self.accuracy
        d['batch'] = self.batch
        d['val_acc_per_epoch'] = self.val_acc_per_epoch
        d['test_acc_per_epoch'] = self.test_acc_per_epoch
        d['train_acc_per_epoch'] = self.train_acc_per_epoch
        w.write(repr(d))
        w.close()

    def cal_accuracy_in_test_set(self):
        cnt = 0
        predictions = self.model.predict(self.test_images)
        for i in range(len(self.test_images)):
            target = np.argmax(predictions[i])
            label = np.argmax(self.test_labels[i])
            if target == label:
                cnt += 1
        return cnt / len(self.test_images)

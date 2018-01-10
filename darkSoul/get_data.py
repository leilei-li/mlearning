import numpy as np


def get_dataset(filename):
    f = open(filename)
    line = f.readline()
    line = f.readline()
    dataSet = []
    while (line):
        data = line.strip().split(' ')
        dd = []
        for d in data:
            dd.append(float(d))
        line = f.readline()
        y = int(float(line.strip()))
        dd.append(y)
        dataSet.append(dd)
        line = f.readline()
    # dataSet = random.sample(dataSet, number)
    batch_x = []
    batch_y = []
    for data in dataSet:
        x = data[:len(data) - 1]
        print(len(data), len(x))
        y = data[len(data) - 1:][0]
        print(y)
        batch_x.append(np.array(x))
        if y == 1:
            batch_y.append(np.array([1, 0, 0]))
        elif y == 0:
            batch_y.append(np.array([0, 1, 0]))
        elif y == -1:
            batch_y.append(np.array([0, 0, 1]))
    return np.array(batch_x), np.array(batch_y)


# get_dataset('attack_training_data.train')
a=[1,2,3]
print(a[2])
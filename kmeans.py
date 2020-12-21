import numpy as np
import tensorflow as tf
from pca import pca,readData

def distance(e1, e2):
    return np.sqrt((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)

def means(arr):
    return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])

def farthest(k_arr, arr):
    f = [0, 0]
    max_d = 0
    for e in arr:
        d = 0
        for i in range(k_arr.__len__()):
            d = d + np.sqrt(distance(k_arr[i], e))
        if d > max_d:
            max_d = d
            f = e
    return f

def closest(a, arr):
    c = arr[1]
    min_d = distance(a, arr[1])
    arr = arr[1:]
    for e in arr:
        d = distance(a, e)
        if d < min_d:
            min_d = d
            c = e
    return c


if __name__=="__main__":
    arr = np.random.randint(100, size=(100, 1, 2))[:, 0, :]
    data = readData()
    pca_data = tf.constant(np.reshape(data, (data.shape[0], -1)), dtype=tf.float32)
    pca_data = pca(pca_data, dim=2)
    arr = pca_data

    m = 4
    r = np.random.randint(arr.__len__() - 1)
    k_arr = np.array([arr[r]])
    cla_arr = [[]]
    for i in range(m-1):
        k = farthest(k_arr, arr)
        k_arr = np.concatenate([k_arr, np.array([k])])
        cla_arr.append([])

    n = 2
    cla_temp = cla_arr
    for i in range(n):
        for e in arr:
            ki = 0
            min_d = distance(e, k_arr[ki])
            for j in range(1, k_arr.__len__()):
                if distance(e, k_arr[j]) < min_d:
                    min_d = distance(e, k_arr[j])
                    ki = j
            cla_temp[ki].append(e)
        for k in range(k_arr.__len__()):
            if n - 1 == i:
                break
            k_arr[k] = means(cla_temp[k])
            cla_temp[k] = []

    print(len(cla_temp))
    print("中心点的坐标为：\n",cla_temp)
    print(cla_temp)
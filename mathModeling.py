import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import csv
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

np.random.seed(1)
liA2 = list()
liCost = list()
literation = 20000
def readData():
    lix = list()
    liy = list()
    f = open('wjq_train_set.csv', 'rt').readlines()
    d = csv.reader(f)
    for line in d:
        linflo = list()
        for ite in line:
            linflo.append(float(ite))
        lix.append(linflo[:-1])
        liy.append(linflo[-1])
        pass
    lix_test = lix[1200:5983]
    lix_train = lix[0:1200]
    liy_test = liy[1200:5983]
    liy_train = liy[0:1200]
    x = np.mat(lix_train).T
    y = np.mat(liy_train)
    x_test = np.mat(lix_test).T
    y_test = np.mat(liy_test)
    return (x,y,x_test,y_test)

def readPredictData():
    li = list()
    f = open('wjq_test_set.csv', 'rt').readlines()
    d = csv.reader(f)
    for line in d:
        linflo = list()
        for ite in line:
            linflo.append(float(ite))
        li.append(linflo)
        pass
    x = np.mat(li).T
    return x
X,Y,X_test,Y_test = readData()
X_pre = readPredictData()
#print(X)
#print(Y)
#plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
#plt.show()
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)
# plot_decision_boundary(lambda x: clf.predict(x), X, Y) #绘制决策边界
# plt.title("Logistic Regression") #图标题
# plt.show()
# LR_predictions  = clf.predict(X.T) #预测结果
# print ("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
# 		np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) +
#        "% " + "(正确标记的数据点所占的百分比)")

def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 29
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)

# print("=========================测试layer_sizes=========================")
# X_asses , Y_asses = layer_sizes_test_case()
# print(X_asses)
# print(Y_asses)
# (n_x,n_h,n_y) =  layer_sizes(X_asses,Y_asses)
# print("输入层的节点数量为: n_x = " + str(n_x))
# print("隐藏层的节点数量为: n_h = " + str(n_h))
# print("输出层的节点数量为: n_y = " + str(n_y))

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.rand(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))

    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (W2.shape == (n_y,n_h))
    assert (b2.shape == (n_y,1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

# print("=========================测试initialize_parameters=========================")
# n_x , n_h , n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x , n_h , n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    #print("W2",W2)
    #print("A1",A1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    liA2.append(A2.tolist()[0])

    assert(A2.shape == (1,X.shape[1]))
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }

    return (A2,cache)

#测试forward_propagation
# print("=========================测试forward_propagation=========================")
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z1"]), np.mean(cache["A1"]), np.mean(cache["Z2"]), np.mean(cache["A2"]))

def compute_cost(A2,Y,parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    logprobs = np.multiply(np.log(A2),Y) + np.multiply((1 - Y),np.log(1 - A2))
    cost = - np.sum(logprobs)/m
    cost = float(np.squeeze(cost))

    assert (isinstance(cost,float))

    return cost

# print("=========================测试compute_cost=========================")
# A2 , Y_assess , parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2,Y_assess,parameters)))

def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2,A1.T)
    db2 = (1 / m) * np.sum(dZ2,axis=1)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1 - np.power(A1,2))
    dW1 = (1 / m) * np.dot(dZ1,X.T)
    db1 = (1 / m) * np.sum(dZ1,axis=1)
    grads = {
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }

    return grads

# #测试backward_propagation
# print("=========================测试backward_propagation=========================")
# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
#
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters,grads,learning_rate):
    W1,W2 = parameters["W1"],parameters["W2"]
    b1,b2 = parameters["b1"],parameters["b2"]

    dW1,dW2 = grads["dW1"],grads["dW2"]
    db1,db2 = grads["db1"],grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}

    return parameters

# #测试update_parameters
# print("=========================测试update_parameters=========================")
# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
#
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def nn_model(X,Y,n_h,num_iterations,learning_rate,print_cost=True):
    np.random.seed(5)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        liCost.append(cost)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost:
            if i%100 == 0:
                print("第",i,"次循环的成本为： " + str(cost))
                predictions = predict(parameters, X_test)
                print('准确率: %d' % float(
                    (np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size) * 100) + '%')
                pass
            pass
        pass

    return parameters

# 测试nn_model
# print("=========================测试nn_model=========================")
# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters)
    print(len(liA2))
    predictions = np.round(A2)

    return predictions

#测试predict
# print("=========================测试predict=========================")
#
# parameters, X_assess = predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("预测的平均值 = " + str(np.mean(predictions)))

parameters = nn_model(X,Y,n_h=29,num_iterations=literation,learning_rate=0.01,print_cost=True)

#绘制边界
#plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
#plt.title("Decision Boundary for hidden layer size " + str(4))
#plt.show()

predictions = forward_propagation(X_pre,parameters)
for item in liA2[-1]:
    print(item)


# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=0.01,print_cost=True)
# plt.plot(liCost,label = 'LR=0.01')
# liCost = []
#
# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=0.03,print_cost=True)
# plt.plot(liCost,label = 'LR=0.03')
# liCost = []
#
# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=0.09,print_cost=True)
# plt.plot(liCost,label = 'LR=0.09')
# liCost = []
#
# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=0.27,print_cost=True)
# plt.plot(liCost,label = 'LR=0.27')
# liCost = []
#
# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=0.81,print_cost=True)
# plt.plot(liCost,label = 'LR=0.81')
# liCost = []
#
# nn_model(X,Y,n_h=13,num_iterations=literation,learning_rate=2.73,print_cost=True)
# plt.plot(liCost,label = 'LR=2.73')
# liCost = []
# plt.ylabel('Cost')
# plt.xlabel('times')
# plt.title('Neural network')
# plt.legend()
# plt.show()

#print(parameters['W1'])
#print(X)

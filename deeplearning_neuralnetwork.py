import numpy as np

np.random.seed(1)

def relu(x):
    return (x>0) * x #возвращает x, если x>0 иначе 0

def relu2deriv(output):
    return output>0 # возвращает 1 если output>0 иначе 0

alpha = 0.2
hidden_size = 4

streerlights = np.array( [[1, 0, 1],
                          [0, 1, 1],
                          [0, 0, 1],
                          [1, 1, 1] ] )

walk_vs_stop = np.array([[1, 1, 0, 0]]).T #T нужна для преобразования строка = столбец

weights_0_1 = 2*np.random.random((3, hidden_size)) - 1 #создает матрицу 3х4
weights_1_2 = 2*np.random.random((hidden_size, 1)) - 1 #создает матрицу 4х1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streerlights)):
        layer_0 = streerlights[0]
        layer_1 = relu(np.dot(layer_0, weights_0_1))  #входной пример layer_0 умножается на матрицу весов weights_0_1
        #                                             и применяется функция активации relu
        layer_2 = np.dot(layer_1, weights_1_2)     #выходной пример layer_1 умножается на матрицу весов weights_1_2

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)
        layer_2_delta = (walk_vs_stop[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)*relu2deriv(layer_1)

        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if (iteration % 10 == 9):
        print("Error")


# This generates the results of the bechmarking code

import Benchmarking


"""
Here are possible combinations of benchmarking user could try.
不同量子门：实现的文件在unitary.py中
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']

各量子门对应的参数量
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

数据编码方式
Encodings: ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact', 'pca32-1', 'autoencoder32-1',
            'pca16-1', 'autoencoder16-1', 'pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1']
数据集
dataset: 'mnist' or 'fashion_mnist'
选择电路，ansatz
circuit: 'QCNN' or 'Hierarchical'
损失函数
cost_fn: 'mse' or 'cross_entropy'
注意：mse对应binary = True
     cross_entropy对应binary = False
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

Unitaries = ['U_SU4', 'U_SU4_1D', 'U_SU4_no_pooling', 'U_9_1D']
U_num_params = [15, 15, 15, 2]
Encodings = ['resize256']
dataset = 'fashion_mnist'
classes = [0, 1]
circuit = 'QCNN'
cost_fn = 'cross_entropy'
binary = False


Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit=circuit, cost_fn=cost_fn, binary=binary)


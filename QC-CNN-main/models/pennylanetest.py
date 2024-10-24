import pennylane as qml
import torch
import torch.nn as nn
import time


def my_quantum_function(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(y, wires=1)
    return qml.expval(qml.PauliZ(1))


'''
加载设备
需要使用gpu加速的话需要用'lightning.gpu' 但是这个设备适合多量子比特的情况 一般就用'default.qubit.torch'
使用pip install pennylane-lightning[gpu]下载设备
使用pip install cuquantum-python下载量子cuda环境
wires是连续整数标签寻址的电线数量 一般为qbit数量 也可以用列表wires=['aux', 'q1', 'q2']来自定义
shots为电路评估次数
'''
# dev = qml.device('default.qubit', wires=2, shots=1000)
# dev = qml.device("lightning.gpu", wires=2)
# dev = qml.device("default.qubit.torch", wires=2, torch_device="cuda:0")


###########分界线##############################################################
# dev = qml.device('default.qubit', wires=4)
# dev = qml.device('default.qubit.torch', wires=16, torch_device='cuda')
dev = qml.device("lightning.gpu", wires=4)

'''
adjoint支持lightning.gpu
'''
@qml.qnode(dev, diff_method="adjoint")
# @qml.qnode(dev, diff_method="adjoint")
# @qml.qnode(dev)
def circuit(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(4), rotation='Y')
    for i in range(4):
        qml.RX(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        weight_shapes = {"weights": (4)}
        qnode = qml.QNode(circuit, dev, interface='torch')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, X):
        X = self.ql1(X)
        X = torch.nn.functional.sigmoid(X)

        return X


torch_cuda = torch.device("cuda")
# torch_cuda = torch.device("cpu")
inputs = torch.rand(10, 4)
labels = torch.randint(0, 2, (10,))

network = Net()
criterion = nn.BCELoss()  # loss function
optimizer = torch.optim.SGD(network.parameters(), lr=0.01)  # optimizer

inputs.to(torch_cuda)
labels.to(torch_cuda)
network.to(torch_cuda)
epochs = 1000
for epoch in range(epochs):
    t1 = time.time()
    tr_loss = 0
    labels = labels.type(torch.FloatTensor)
    labels = labels.to(torch_cuda)
    optimizer.zero_grad()
    outputs = network(inputs)
    outputs = outputs.to(torch_cuda)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    cpu_loss = loss.to("cpu")
    tr_loss = cpu_loss.data.numpy()

    print("loss:{:.3f},time:{:.3f}".format(tr_loss, time.time() - t1))

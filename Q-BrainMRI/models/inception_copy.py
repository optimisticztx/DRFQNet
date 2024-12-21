import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from math import ceil
from math import pi

torch.manual_seed(0)

n_qubits = 4
n_layers = 1
n_class = 3
n_features = 196
image_x_y_dim = 14
kernel_size = n_qubits
stride = 2

# dev = qml.device("default.qubit", wires=n_qubits)
# dev = qml.device('default.qubit.torch', wires=n_qubits, torch_device='cuda')
dev = qml.device('default.qubit', wires=n_qubits)
# dev = qml.device("lightning.gpu", wires=n_qubits)

# 4qbit对4*4图像进行卷积 单比特编码多数据 最后纠缠+参数
@qml.qnode(dev)
def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)

    # 输入编码部分
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        for i in range(var_per_qubit):
            idx = qub * var_per_qubit + i

            if idx < len(inputs):
                if encoding_gates[i] == 'RZ':
                    qml.RZ(inputs[idx], wires=qub)
                elif encoding_gates[i] == 'RY':
                    qml.RY(inputs[idx], wires=qub)
            else:  # load nothing
                pass

    # 参数化层部分
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
            # qml.CNOT(wires = [i, (i + 1) % n_qubits])
        for j in range(n_qubits, 2 * n_qubits):
            qml.RY(weights[l, j], wires=j % n_qubits)

    # 批量返回每个qubit （为测量）
    _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return _expectations


'''
卷积操作的手动实现可以参考：
https://blog.csdn.net/m0_37567738/article/details/132397002
优化可以参考chatGPT
'''


# 字典的shape（1，2*4）=（1， 8）
# 这种显示循环效率较低
class Quanv2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, 2 * n_qubits)}
        # diff_method='adjoint','parameter-shift','best'
        qnode = qml.QNode(circuit, dev, interface='torch', diff_method='best')
        self.ql1 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, X):
        assert len(X.shape) == 4  # (bs, c, w, h)
        bs = X.shape[0]
        XB = []
        for b in range(0, bs):
            XL = []
            for i in range(0, X.shape[2] - 2, stride):
                for j in range(0, X.shape[3] - 2, stride):
                    tt = torch.flatten(X[b, :, i:i + kernel_size, j:j + kernel_size], start_dim=0)
                    xx = self.ql1(tt)
                    XL.append(xx)
            XB.append(torch.cat(XL, dim=0).view(4, 6, 6))
        X = torch.stack(XB, dim=0)
        return X




# 一个bit对应一个通道 所以4通道


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branchClassic_1 = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)
        self.branchClassic_2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)
        # self.branchClassic_3 = nn.Conv2d(1, 4, kernel_size=4, stride=2)
        self.branchQuantum = Quanv2d(kernel_size=4, stride=2)

    def forward(self, x):
        classic = self.branchClassic_1(x)
        # print("b1 shape:{}".format(classic.shape))# torch.Size([bs, 4, 14, 14])

        classic = self.branchClassic_2(classic)
        # print("b2 shape:{}".format(classic.shape))# torch.Size([bs, 8, 6, 6])
        # classic2 = self.branchClassic_3(x)
        quantum = self.branchQuantum(x)
        # print("quantum1 shape:{}".format(quantum.shape))# torch.Size([bs, 4, 6, 6])
        outputs = [classic, quantum]
        # outputs = [classic, classic2]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.incep = Inception(in_channels=1)
        #8通道（2层卷积后） + 4通道（1层量子卷积）
        self.fc1 = nn.Linear(12 * 6 * 6, n_class*2)
        # self.fc1 = nn.Linear(8 * 6 * 6, n_class*2)
        self.fc2 = nn.Linear(n_class*2, n_class)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 1, 14, 14)
        x = self.incep(x)
        x = self.lr(x)

        x = x.view(bs, -1)
        x = self.lr(self.fc1(x))
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch._tensor as tensor
import torch.nn.functional as F
import pennylane as qml
from math import ceil
from math import pi

torch.manual_seed(0)

n_qubits = 8
n_layers = 2
n_class = 4
kernel_size = n_qubits
stride = 4

dev = qml.device("default.qubit", wires=n_qubits)
# dev = qml.device("lightning.qubit", wires=n_qubits)
# dev = qml.device("lightning.gpu", wires=n_qubits)

@qml.qnode(device=dev,interface='torch',diff_method='best')
def circuit(inputs, weights):
    var_per_qubit = int(len(inputs) / n_qubits) + 1
    encoding_gates = ['RZ', 'RY'] * ceil(var_per_qubit / 2)  # 编码使用的量子门

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
            if (l+1) % 2 == 0:# 第二层用cry
                qml.CRY(weights[l, i], wires=[i, (i + 1) % n_qubits])
            else:# 第一层用crx
                qml.CRX(weights[l, i], wires=[i, (i + 1) % n_qubits])


    # 添加测量
    # _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits) if i%2==0]
    _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return _expectations



class Quanv2d(nn.Module):
    def __init__(self, kernel_size=None, stride=None):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers, n_layers * n_qubits)}
        self.ql1 = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, X):
        # assert len(X.shape) == 4  # (bs, c, w, h)
        bs = X.shape[0]
        XB = []
        for b in range(0, bs):
            # XL = []
            # for i in range(0, X.shape[2] - stride + 1, stride):
            #     for j in range(0, X.shape[3] - stride + 1, stride):
            #         tt = torch.flatten(X[b, :, i:i + kernel_size, j:j + kernel_size], start_dim=0)
            #         xx = self.ql1(tt)
            #         XL.append(xx)
            XL = self.ql1(X[b,:])
            # print("XL shape:",XL.shape)
            # XB.append(torch.cat(XL, dim=0).view(-1))
            XB.append(XL)
            # print("XL:",XL)
        X = torch.stack(XB, dim=0)
        # print("X:",X)
        return X


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.branchClassic_1 = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)
        self.branchClassic_2 = nn.Conv2d(4, 8, kernel_size=4, stride=4)

        self.branchQuantum = Quanv2d(kernel_size=4, stride=4)

    def forward(self, x):
        # classic = self.branchClassic_1(x)
        # classic = self.branchClassic_2(classic)
        # print("classic shape:",classic.shape)
        quantum = self.branchQuantum(x)
        # print("quantum shape:",quantum.shape)
        # outputs = [classic, quantum]
        # return torch.cat(outputs, dim=1)
        return quantum


class Net(nn.Module):
    def __init__(self,preNet=None):
        super(Net, self).__init__()
        self.preNet = preNet
        self.quantumCircuit = Inception(in_channels=3)# 3*224*224
        # self.fc1 = nn.Linear(12 * 56 * 56, n_class*2)# 8经典卷积通道+4量子卷积通道
        self.fc2 = nn.Linear(8, n_class)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self, x):
        # print("X shape:",x.shape)
        bs = x.shape[0]
        x = x.view(bs, 3, 224, 224)
        x = self.preNet(x)# 输出结果为8
        # print("X preNet shape:",x.shape)
        x = self.lr(x)
        x = self.quantumCircuit(x)
        # print("X quantumCircuit shape:",x.shape)
        x = self.lr(x)
        x = x.view(bs, -1)
        # x = self.lr(self.fc1(x))
        x = self.fc2(x)
        return x

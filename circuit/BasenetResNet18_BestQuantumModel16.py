import math

import torch
import torch.nn as nn
import torch._tensor as tensor
import torch.nn.functional as F
import pennylane as qml
from math import ceil
from math import pi

torch.manual_seed(3407)

in_shape = 16
n_qubits = 8
n_class = 4
n_layers_crz = 1
n_layers_crx = 8
kernel_size = n_qubits

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(device=dev,interface='torch',diff_method='best')
def circuit(inputs, weights):
    encoding_gates = ['RZ', 'RY'] # DRAE gates

    # DRAE module
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub) # init
        for i in range(2):
            idx = qub * 2 + i
            if idx < len(inputs):
                if encoding_gates[i] == 'RZ':
                    qml.RZ(inputs[idx], wires=qub)
                elif encoding_gates[i] == 'RY':
                    qml.RY(inputs[idx], wires=qub)
            else:  # load nothing
                pass

    # FQE module (Phase Entanglement)
    for l in range(n_layers_crz):
        for i in range(n_qubits):
            qml.CRZ(weights[l, i], wires=[i, (i + 1) % n_qubits])
    # FQE module (Amplitude Entanglement)
    for i in range(n_qubits):
        for j in range(n_qubits):
            if j == i:
                continue
            else:
                qml.CRX(weights[n_layers_crz + i, j], wires=[i, j])

    # Measurement
    _expectations = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return _expectations



class Quantum(nn.Module):
    def __init__(self):
        super(Quanv2d, self).__init__()
        weight_shapes = {"weights": (n_layers_crz + n_qubits, (n_layers_crz + n_qubits) * n_qubits)}
        self.ql1 = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, X):
        bs = X.shape[0]
        XB = []
        for b in range(0, bs):
            XL = self.ql1(X[b,:])
            XB.append(XL)
        X = torch.stack(XB, dim=0)
        return X


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branchQuantum = Quantum()

    def forward(self, x):

        quantum = self.branchQuantum(x)
        return quantum


class Net(nn.Module):
    def __init__(self,preNet=None, n_class=4):
        super(Net, self).__init__()
        self.preNet = preNet
        self.quantumCircuit = Inception(in_channels=3)
        self.fc1 = nn.Linear(8, n_class)
        self.lr = nn.LeakyReLU(0.1)

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, 3, 224, 224)
        x = self.preNet(x)# 16
        x = self.lr(x)
        x = self.quantumCircuit(x)
        x = self.lr(x)
        x = x.view(bs, -1)
        x = self.fc1(x)
        return x

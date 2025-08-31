import pennylane as qml
n_layers = 3

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for l in range(2):
        for i in range(4):
            if l == 0:  # 第一层用rx
                qml.RX(inputs[i], wires=i)
            else:  # 第二层用rz
                qml.RZ(weights[l, i], wires=i)
    qml.CRX(weights[2, 2], wires=[3, 2])
    qml.CRX(weights[2, 1], wires=[2, 1])
    qml.CRX(weights[2, 0], wires=[1, 0])


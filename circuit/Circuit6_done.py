import pennylane as qml

n_layers = 7

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.RX(inputs[qub], wires=qub)
    for qub in range(n_qubits):
        qml.RZ(weights[0, qub], wires=qub)
    qml.CRX(weights[1, 2], wires=[3, 2])
    qml.CRX(weights[1, 1], wires=[3, 1])
    qml.CRX(weights[1, 0], wires=[3, 0])

    qml.CRX(weights[2, 3], wires=[2, 3])
    qml.CRX(weights[2, 1], wires=[2, 1])
    qml.CRX(weights[2, 0], wires=[2, 0])

    qml.CRX(weights[3, 3], wires=[1, 3])
    qml.CRX(weights[3, 2], wires=[1, 2])
    qml.CRX(weights[3, 0], wires=[1, 0])

    qml.CRX(weights[4, 3], wires=[0, 3])
    qml.CRX(weights[4, 2], wires=[0, 2])
    qml.CRX(weights[4, 1], wires=[0, 1])

    for qub in range(n_qubits):
        qml.RX(weights[5, qub], wires=qub)
    for qub in range(n_qubits):
        qml.RZ(weights[6, qub], wires=qub)








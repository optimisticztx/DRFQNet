import pennylane as qml
n_layers = 5

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.RX(inputs[qub], wires=qub)
    for qub in range(n_qubits):
        qml.RZ(weights[0, qub], wires=qub)
    qml.CRX(weights[1, 0], wires=[1, 0])
    qml.CRX(weights[1, 2], wires=[3, 2])
    for qub in range(n_qubits):
        qml.RX(weights[2, qub], wires=qub)
    for qub in range(n_qubits):
        qml.RZ(weights[3, qub], wires=qub)
    qml.CRX(weights[4, 1], wires=[2, 1])








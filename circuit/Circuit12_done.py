import pennylane as qml
n_layers = 3

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.RY(inputs[qub], wires=qub)
    for qub in range(n_qubits):
        qml.RZ(weights[0, qub], wires=qub)
    qml.CZ(wires=[3, 2])
    qml.CZ(wires=[1, 0])
    qml.RY(weights[1, 1], wires=1)
    qml.RY(weights[1, 2], wires=2)
    qml.RZ(weights[2, 1], wires=1)
    qml.RZ(weights[2, 2], wires=2)
    qml.CZ(wires=[2, 1])





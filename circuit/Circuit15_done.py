import pennylane as qml
n_layers = 1

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.RY(inputs[qub], wires=qub)
    qml.CNOT(wires=[3, 0])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[0, 1])
    for qub in range(n_qubits):
        qml.RY(weights[0, qub], wires=qub)
    qml.CNOT(wires=[3, 2])
    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 1])






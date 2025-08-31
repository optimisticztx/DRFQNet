import pennylane as qml
n_layers = 3

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# 参数化层部分
@qml.qnode(device=dev, interface='torch', diff_method='best')
def circuit(inputs, weights):
    for qub in range(n_qubits):
        qml.RY(inputs[qub], wires=qub)
    qml.CRZ(weights[0, 0], wires=[3, 0])
    qml.CRZ(weights[0, 3], wires=[2, 3])
    qml.CRZ(weights[0, 2], wires=[1, 2])
    qml.CRZ(weights[0, 1], wires=[0, 1])
    for qub in range(n_qubits):
        qml.RY(weights[1, qub], wires=qub)
    qml.CRZ(weights[2, 2], wires=[3, 2])
    qml.CRZ(weights[2, 3], wires=[0, 3])
    qml.CRZ(weights[2, 0], wires=[1, 0])
    qml.CRZ(weights[2, 1], wires=[2, 1])






# QOSF_project

这是用于经典数据分类的量子卷积神经网络的实现 (<https://arxiv.org/abs/2108.00661>)。它使用 Pennylane 软件 ([https://pennylane.ai](https://pennylane.ai/)) 对 MNIST 和 Fashion MNIST 数据集进行分类。

### 1. 数据准备

**"data.py "**：加载经过经典预处理的数据集（MNIST 或 Fashion MNIST）。"feature_reduction "参数包含预处理方法和预处理数据维度的信息。

### 2. QCNN 电路

**"QCNN_circuit.py"**: QCNN 函数实现了量子卷积神经网络。"QCNN_structure" 是标准结构，包含迭代卷积层和池化层。"QCNN_structure_without_pooling "只有卷积层。"QCNN_1D_circuit "去掉了第一个和最后一个量子比特之间的连接。(**"Hierarchical.py "**: 类似于分层量子分类器结构，<https://www.nature.com/articles/s41534-018-0116-9>)

**"unitary.py "**：包含卷积层和池化层使用的所有单元解析。

**"embedding.py "**：展示了如何将经典数据初步嵌入 QCNN 电路。主要有五种嵌入方式： 幅度嵌入、角度嵌入、紧凑角度嵌入（"Angle-compact"）、混合直接嵌入（"Amplitude-Hybrid"）、混合角度嵌入（"Angulr-Hybrid"）。

**"Angular_hybrid.py "**：包含在**"embedding.py "**中使用的混合角度嵌入结构。

混合直接嵌入（Hybrid Direct Embedding）和混合角度嵌入（Hybrid Angle Embedding）根据嵌入块中的量子比特数量和嵌入排列方式而有所不同。例如，"Amplitude-Hybrid4-1 "将 16 个经典数据嵌入排列为[[0,1,2,3], [4,5,6,7]]的 4 个量子位中。

### 3. QCNN 训练

**"Training.py "**：训练量子回路（QCNN 或 Hierarchical）。默认情况下，它使用内斯托夫动量优化器（nesterov momentum optimizer）训练 200 次，批量为 25 次。MSE 损失和交叉熵损失均可用于电路训练。测试不同 QCNN 结构时，需要调整总参数数（total_params）。

### 4. 基准测试

**"Benchmarking.py "**：针对给定的数据集、单元解析和编码/嵌入方法训练量子电路。保存训练损失历史和训练后测试数据的准确性。Encoding_too_Embedding 函数将编码（经典预处理特征缩减）转换为嵌入（经典数据嵌入量子电路）。

二进制： 真 "使用 1 和 -1 标签，而 "假 "使用 1 和 0 标签。使用交叉熵代价函数时，始终使用 "False"。使用 mse 成本函数时，论文结果中使用 "True"，但也可以使用 "False"。


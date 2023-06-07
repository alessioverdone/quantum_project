import torch.nn as nn
import pennylane as qml


def prepare_network_for_t_learning(net, n_layers, n_qubits, qnode, **kwargs):
    weight_shapes = {"weights": (n_layers, n_qubits)}
    print(weight_shapes)
    freeze_parameters = kwargs.get('freeze_parameters', True)
    in_j_layer = kwargs.get('in_j_layer', 7)
    out_j_layer = kwargs.get('out_j_layer', 2)
    if freeze_parameters == True:
        for param in net.parameters():
            param.requires_grad = False
    net_name = net.__class__.__name__
    if net_name == 'DenseNet':
        # Getting the number of output classes
        output_classes = kwargs.get('output_classes', 6)
        num_ftrs = net.classifier.in_features

        junc_layer_1 = nn.Linear(num_ftrs, n_qubits)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        junc_layer_2 = nn.Linear(n_qubits, output_classes)

        # Adapating the last layer of the network to the specific classification task
        net.classifier = nn.Sequential(junc_layer_1, qlayer, junc_layer_2)
    elif net_name == 'ResNet':
        # Getting the number of output classes
        output_classes = kwargs.get('output_classes', 6)
        num_ftrs = net.fc.in_features

        junc_layer_1 = nn.Linear(num_ftrs, n_qubits)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        junc_layer_2 = nn.Linear(n_qubits, output_classes)

        # Adapating the last layer of the network to the specific classification task
        net.fc = nn.Sequential(junc_layer_1, qlayer, junc_layer_2)
    elif net_name == 'VGG':
        output_classes = kwargs.get('output_classes', 6)
        num_ftrs = net.classifier[6].in_features

        junc_layer_1 = nn.Linear(num_ftrs, n_qubits)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        junc_layer_2 = nn.Linear(n_qubits, output_classes)

        net.classifier[6] = nn.Sequential(junc_layer_1, qlayer, junc_layer_2)
    elif net_name == 'EfficientNet':
        output_classes = kwargs.get('output_classes', 6)
        num_ftrs = net.classifier.fc.in_features
        # Adapating the last layer of the network to the specific classification task

        junc_layer_1 = nn.Linear(num_ftrs, n_qubits)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        junc_layer_2 = nn.Linear(n_qubits, output_classes)

        net.classifier.fc = nn.Sequential(junc_layer_1, qlayer, junc_layer_2)
    elif net_name == "Inception3":
        output_classes = kwargs.get('output_classes', 6)
        num_ftrs = net.fc.in_features
        # Adapating the last layer of the network to the specific classification task

        junc_layer_1 = nn.Linear(num_ftrs, n_qubits)
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        junc_layer_2 = nn.Linear(n_qubits, output_classes)

        net.fc = nn.Sequential(junc_layer_1, qlayer, junc_layer_2)
        net.aux_logits = False
    num_params = 0
    num_params += sum(param.numel() for param in net.parameters() if param.requires_grad)
    print('You requested the train of: ', net_name)
    print('Number of parameters:', num_params)
    return net


def get_quantum_layer(qnode_id):

    if qnode_id == 1:
        # Quantum architecture #1
        n_qubits = 4  # 8Qubits
        n_layers = 2
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RX)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # ------------------------------------------------------------------------------------------------------------------

    # Quantum architecture #2
    elif qnode_id == 2:
        n_qubits = 4  # 8Qubits
        n_layers = 1
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # ------------------------------------------------------------------------------------------------------------------

    # Quantum architecture #3
    elif qnode_id == 3:
        n_qubits = 4  # 8Qubits
        n_layers = 3
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            i = 0
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # ------------------------------------------------------------------------------------------------------------------

    # Quantum architecture #4
    # Creating the quantum layer
    elif qnode_id == 4:
        n_qubits = 4  # 8Qubits
        n_layers = 3
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def qnode(inputs, weights):
            i = 0
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            for l in range(0, n_layers):
                for q in range(0, n_qubits):
                    qml.RY(weights[l, q], q)
            for l in range(0, n_layers):
                for q in range(0, n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    # ------------------------------------------------------------------------------------------------------------------

    # Quantum architecture #5
    # Creating the quantum layer
    elif qnode_id == 5:
        n_qubits = 4  # 8Qubits
        n_layers = 4 * 2
        dev = qml.device("lightning.qubit", wires=n_qubits)

        def entanglement(n_qubits, type='linear'):
            if type == 'linear':
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, i + 1])
            if type == 'circular':
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            if type == 'full':
                for i in range(n_qubits - 1):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])

        @qml.qnode(dev)
        def qnode(inputs, weights):
            # qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            # qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            for i in range(len(inputs)):
                qml.RY(inputs[i], wires=i)
            for l in range(int(n_layers / 2)):
                for q in range(n_qubits):
                    qml.RX(weights[l, q], q)
            for l in range(int(n_layers / 2), n_layers):
                for q in range(n_qubits):
                    qml.RY(weights[l, q], q)
                entanglement(n_qubits, 'full')
            # qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RY)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    else:
        raise Exception('No quantum layer selected!')

    # ------------------------------------------------------------------------------------------------------------------

    return qnode

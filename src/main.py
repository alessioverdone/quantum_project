import argparse
from torchvision import models
import torch
from training.solver import Solver
from utils import prepare_network_for_t_learning, get_quantum_layer
import zipfile


def main():
    parser = argparse.ArgumentParser()

    # model_set = {'ResNet50': models.resnet50(pretrained=True),
    #              'DenseNet169': models.densenet169(pretrained=True),
    #              'VGG19': models.vgg19(pretrained=True),
    #              'EfficientNet': torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
    #                                             pretrained=True),
    #              'Inception': torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)}
    model_set = {'ResNet50': models.resnet50(pretrained=True)}

    parser.add_argument("--model_to_test", type=str, default='ResNet50')
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=30)

    parser.add_argument("--ckpt_dir", type=str,
                        default="I:\Il mio Drive\PhD ICT\Theory\quantum\quantum_project\quantum_project_folder\checkpoints")
    parser.add_argument("--path_to_test", type=str,
                        default="/content/gdrive/MyDrive/Progetto Quantum Binucci PhD/QuantumCheckpointFrank/VGGFRA3/landscape_best.pth")
    parser.add_argument("--ckpt_name", type=str, default="landscape")
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--print_every_minibatches", type=int, default=2)

    # if you change image size, you must change all the network channels
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--data_root", type=str,
                        default="I:\Il mio Drive\PhD ICT\Theory\quantum\quantum_project\quantum_project_folder\data")

    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--qnode", type=int, default=1)
    args, unknown = parser.parse_known_args()
    print('Debug!')

    unzip_file = False
    if unzip_file:
        zip_ref = zipfile.ZipFile("/content/gdrive/MyDrive/PhDQuantumProject/archive.zip", 'r')
        zip_ref.extractall("/content/dataset")
        zip_ref.close()

    # Get the quantum layer
    qnode = get_quantum_layer(args.qnode)
    net = prepare_network_for_t_learning(model_set[args.model_to_test], args.n_layers, args.n_qubits, qnode,
                                         freeze_parameters=True)

    print("Test mode" if args.test else "Train Mode")

    solver = Solver(args, net=net)

    if args.test == False:
        solver.fit()
    else:
        net.load_state_dict(torch.load(args.path_to_test))
        solver.load_network(net)
        accuracy = solver.test()
        print(f'Accuracy on the test-set for {net.__class__.__name__} = {accuracy * 100}%')


if __name__ == "__main__":
    main()

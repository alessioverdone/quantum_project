from dataset.intel_scene_classification.dataset import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
import os


class Solver():
    def __init__(self, args, **kwargs):
        # prepare a dataset
        self.train_data = Dataset(train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)

        self.test_data = Dataset(train=False,
                                 data_root=args.data_root,
                                 size=args.image_size)

        lengths = [int(0.8 * len(self.train_data)), len(self.train_data) - int(0.8 * len(self.train_data))]
        self.train_set, self.val_set = torch.utils.data.random_split(self.train_data,
                                                                     lengths,
                                                                     torch.Generator().manual_seed(42))

        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f'Current device: {self.device}')
        ########################################################################
        # Implementing ResNet50 module, used for feature extraction purposes.
        cnn_network = models.resnet50(pretrained=True).to(self.device)

        # Defining the connection part between the classical network and the quantistic fully connected network
        num_ftrs = cnn_network.fc.in_features
        cnn_network.fc = nn.Linear(num_ftrs, args.n_qubits).to(self.device)
        clayer_1 = torch.nn.Linear(args.n_qubits, 6).to(self.device)
        # clayer_2 = torch.nn.Linear(10,6).to(self.device)

        # Defining the quantum layer
        # Loading network to train/test. By default, the script load the pre-trained ResNet50
        self.net = kwargs.get('net', models.resnet50(pretrained=True)).to(self.device)
        print(f'You requested the training of: {self.net.__class__.__name__}')
        ########################################################################

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters()
                                      , lr=args.lr)  # torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def fit(self):
        args = self.args
        max_accuracy = 0
        for epoch in range(args.max_epochs):
            self.net.train()
            for step, inputs in enumerate(self.train_loader):

                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)
                pred = self.net(images)
                loss = self.loss_fn(pred, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                # train_acc = self.evaluate(self.train_data)
                # test_acc = self.evaluate(self.test_data)
                if (step + 1) % args.print_every_minibatches == 0:
                    print("Epoch [{}/{}] Batch [{}/{}] Loss: {:.3f}".
                          format(epoch + 1, args.max_epochs, step, len(self.train_loader), loss.item()))

                # print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                #      format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))
                # print("Epoch [{}/{}] Loss: {:.3f} ".
                # format(epoch + 1, args.max_epochs, loss.item()))
            if (epoch + 1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.val_set)

            if test_acc > max_accuracy:
                max_accuracy = test_acc
                self.save(args.ckpt_dir, args.ckpt_name, 'best')

            print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                  format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))

    def evaluate(self, data):
        args = self.args
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            num_workers=1,
                            shuffle=False)

        self.net.eval()
        num_correct, num_total = 0, 0

        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device)

                outputs = self.net(images)
                _, preds = torch.max(outputs.detach(), 1)

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total

    def load_network(self, net):
        self.net = net

    def test(self):
        return self.evaluate(self.test_data)

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

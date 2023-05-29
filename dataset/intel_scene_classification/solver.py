import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net

from dataset import Dataset


class Solver():
    def __init__(self, args):
        # prepare a dataset
        self.train_data = Dataset(train=True,
                                  data_root=args.data_root,
                                  size=args.image_size)


        self.test_data = Dataset(train=False,
                                 data_root=args.data_root,
                                 size=args.image_size)

        lengths = np.array([int(0.8*len(self.train_data)), len(self.train_data) - int(0.8 * len(self.train_data))])
        self.train_set, self.val_set = torch.utils.data.random_split(self.train_data,
                                                           lengths,
                                                           torch.Generator().manual_seed(42))

        self.train_loader = DataLoader(dataset=self.train_set,
                                       batch_size=args.batch_size,
                                       num_workers=4,
                                       shuffle=True, drop_last=True)

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = Net().to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.args = args

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    def fit(self):
        args = self.args

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

                # print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                #      format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))
                #print("Epoch [{}/{}] Loss: {:.3f} ".
                      #format(epoch + 1, args.max_epochs, loss.item()))
            if (epoch + 1) % args.print_every == 0:
                train_acc = self.evaluate(self.train_data)
                test_acc = self.evaluate(self.val_set)

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                      format(epoch + 1, args.max_epochs, loss.item(), train_acc, test_acc))

                self.save(args.ckpt_dir, args.ckpt_name, epoch + 1)

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

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, global_step))
        torch.save(self.net.state_dict(), save_path)

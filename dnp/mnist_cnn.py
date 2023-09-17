import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params_proto import ParamsProto, Proto, Flag
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer):
    from ml_logger import logger
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(Params.device), target.to(Params.device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accuracy = pred.eq(target.view_as(pred)).sum().item()
        logger.store_metrics({'train/loss': loss.item(), 'train/accuracy': accuracy / len(target), })

        # if batch_idx % Params.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))
        #     if Params.dry_run:
        #         break


@torch.no_grad()
def test(model, test_loader):
    from ml_logger import logger
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(Params.device), target.to(Params.device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='mean')
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy = pred.eq(target.view_as(pred)).sum().item()

            logger.store_metrics({'test/loss': loss.item(), 'test/accuracy': accuracy / len(target)})

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


class Params(ParamsProto):
    """MNIST example"""
    dataset_root: str = Proto("/tmp/datasets", env="$DATASETS")

    # Training settings
    seed: int = Proto(1, help='random seed (default: 1)')

    target = "random"  # one of [None, "random"]

    n_epochs: int = Proto(14, help='number of epochs to train (default: 14)')
    batch_size: int = Proto(64, help='input batch size for training (default: 64)')
    test_batch_size: int = Proto(1000, help='input batch size for testing (default: 1000)')
    lr: float = Proto(1.0, metavar='LR', help='learning rate (default: 1.0)')
    gamma: float = Proto(0.7, help='Learning rate step gamma (default: 0.7)')

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    log_interval: int = Proto(1, help='how many batches to wait before logging training status')
    checkpoint_interval: int = None


def main(charts=None, **deps):
    from ml_logger import logger

    Params._update(**deps)
    logger.log_params(Params=vars(Params))

    if charts != False:
        logger.log_text(charts or """
        charts:
        - yKeys: ["train/accuracy/mean", "test/accuracy/mean"]
          xKey: epoch
        - yKeys: ["train/loss/mean"]
          xKey: epoch
        """, filename=".charts.yml", dedent=True)

    np.random.seed(Params.seed)
    torch.manual_seed(Params.seed)

    train_kwargs = {'batch_size': Params.batch_size}
    test_kwargs = {'batch_size': Params.test_batch_size}

    if Params.device == "cuda":
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    d_train = datasets.MNIST(Params.dataset_root, train=True, download=True, transform=transform)
    d_dev = datasets.MNIST(Params.dataset_root, train=False, transform=transform)

    if Params.target == "random":
        original = d_train.targets
        rand_inds = torch.randperm(len(original))
        d_train.targets = original[rand_inds]
        d_train.targets

    train_loader = torch.utils.data.DataLoader(d_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(d_dev, **test_kwargs)

    model = Net().to(Params.device)
    optimizer = optim.Adadelta(model.parameters(), lr=Params.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=Params.gamma)

    for epoch in range(1, Params.n_epochs + 1):

        train(model, train_loader, optimizer)
        # logger.store_metrics(lr=scheduler.get_last_lr())
        test(model, test_loader)
        # scheduler.step()

        if epoch % Params.log_interval == 0:
            logger.log_metrics_summary(key_values={"epoch": epoch})
        if logger.every(Params.checkpoint_interval, key="cp", start_on=1):
            logger.save_torch(model, f"checkpoints/net_{epoch}.pt")
            logger.duplicate(f"checkpoints/net_{epoch}.pt", f"checkpoints/net_last.pt")

    # if Params.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

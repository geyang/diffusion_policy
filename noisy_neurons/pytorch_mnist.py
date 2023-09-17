import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from params_proto import ParamsProto, Proto, Flag


class Args(ParamsProto):
    """PyTorch MNIST Example"""

    net: str = Proto("CNN", help="CNN or MLP")

    dataset_root = Proto(env="$DATASETS", help="path to datasets")
    batch_size = Proto(64, help="input batch size for training (default: 64)")
    test_batch_size = Proto(1000, help="input batch size for testing (default: 1000)")
    epochs = Proto(14, help="number of epochs to train (default: 14)")
    lr = Proto(1.0, help="learning rate (default: 1.0)")
    gamma = Proto(0.7, help="Learning rate step gamma (default: 0.7)")
    no_cuda = Flag(help="disables CUDA training")
    # dry_run = Flag(help="quickly check a single pass")
    seed = Proto(1, help="random seed (default: 1)")
    log_interval = Proto(
        100, help="how many batches to wait before logging training status"
    )

    if torch.backends.mps:
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    save_checkpoint = Flag(
        to_value="checkpoints/net.pt", help="For Saving the current Model"
    )


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 100, 3, 1),
            nn.ReLU(),
            nn.Conv2d(100, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )


class MLP(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main(**kwargs):
    from ml_logger import logger

    Args._update(**kwargs)

    logger.job_started(Args=vars(Args))

    torch.manual_seed(Args.seed)

    dwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST(
        Args.dataset_root, train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(Args.dataset_root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=Args.batch_size, **dwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=Args.test_batch_size, **dwargs
    )

    Net = eval(Args.net)
    model = Net().to(Args.device)
    # x = torch.zeros([1, 1, 28, 28]).to(Args.device)
    # model_train = torch.jit.trace(model.train(), x)
    # model_eval = torch.jit.trace(model.eval(), x)

    optimizer = optim.Adadelta(model.parameters(), lr=Args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=Args.gamma)

    for epoch in range(1, Args.epochs + 1):
        train(Args, model, Args.device, train_loader, optimizer, epoch)
        evaluate(model, Args.device, test_loader)
        scheduler.step()

    if Args.save_checkpoint:
        logger.torch_save(model, Args.save_checkpoint)

    logger.job_completed()


if __name__ == "__main__":
    main()

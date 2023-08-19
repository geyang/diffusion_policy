import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from params_proto import ParamsProto, Proto, Flag


class Args(ParamsProto):
    """PyTorch MNIST Example"""

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

    save_model = Flag(help="For Saving the current Model")
    device = "mps"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # PLU Specific Experimental Flags
    # act_fn_1 = "plu"


# act_fns = dict(
#     sine=lambda x: torch.sin(x),
#     plu=lambda x: torch.stack([torch.abs(x % 2), torch.abs(-x % 2)]).min(dim=0).values
#     - 0.5,
# )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 3, 1)
        self.conv2 = nn.Conv2d(100, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # act_fn_1 = act_fns.get(Args.act_fn_1, None) or getattr(F, Args.act_fn_1)

        # note: A few things are off:
        #   MNIST is probably not the best task, because the color distribution is binary.
        #   ~
        #   1. We do need to increase the number of channels to 3 * 40, judging from experience.
        #   2. Hard to fairly compare speed and computation complexity.
        #   3. The sparsity bias for ReLU to set the output to 0 is missing.
        x = self.conv1(x)
        # x = act_fn_1(10 * x)
        x = F.relu(x)
        x = self.dropout1(x)
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
            # if args.dry_run:
            #     break


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

    train_kwargs = {"batch_size": Args.batch_size}
    test_kwargs = {"batch_size": Args.test_batch_size}

    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST(
        Args.dataset_root, train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(Args.dataset_root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(Args.device)
    optimizer = optim.Adadelta(model.parameters(), lr=Args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=Args.gamma)
    for epoch in range(1, Args.epochs + 1):
        train(Args, model, Args.device, train_loader, optimizer, epoch)
        evaluate(model, Args.device, test_loader)
        scheduler.step()

    if Args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    logger.job_completed()


if __name__ == "__main__":
    main()

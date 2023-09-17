from functools import partial
from random import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from params_proto import ParamsProto
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam
from tqdm import tqdm

from noisy_ntk.models.rff_mlp import MLP, LFF


class Buffer:
    def __init__(self, *data, batch_size=None):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data[0])

    def sample(self, batch_size):
        inds = torch.rand(size=(self.__len__(),)).argsort()
        from more_itertools import chunked

        for batch_inds in chunked(inds, n=batch_size):
            yield [torch.Tensor(d[batch_inds]) for d in self.data]


class Params(ParamsProto):
    seed = 100
    n_layers = 4
    lat_dim = 400
    lr = 1e-4
    batch_size = 32
    n_epochs = 2000_000

    target = "sine"
    target_k = None
    network = "mlp"
    ffn_scale = 10

    checkpoint = None

    log_interval = 10_000
    checkpoint_stops = [100, 1000, 10_000, 20_000, 50_000, 100_000]
    checkpoint_interval = 200_000
    # ntk_interval = 10_000
    plot_interval = 10_000

    metrics_prefix = None

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


@torch.no_grad()
def plot_fit(xs, ys, net, filename=None):
    from ml_logger import logger

    fitted = net(torch.FloatTensor(xs[:, None]).to(Params.device))[:, 0].cpu().numpy()

    plt.title("Target Function")
    plt.plot(xs, ys, color="black", label="target")
    plt.plot(xs, fitted, color="orange", linewidth=4, alpha=0.8, label="fit")
    plt.xlabel("x")
    plt.ylabel(Params.target)
    plt.legend(frameon=False)
    if filename:
        logger.savefig(filename, close=True)


def get_ntk(net, xs):
    grad = []
    out = net(torch.FloatTensor(xs)[:, None].to(Params.device))
    for o in tqdm(out, desc="NTK", leave=False):
        net.zero_grad()
        o.backward(retain_graph=True)
        grad_vec = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None]).cpu().numpy()
        grad.append(grad_vec / np.linalg.norm(grad_vec))

    net.zero_grad()
    grad = np.stack(grad)
    gram_matrix = grad @ grad.T
    return gram_matrix


# net = FFN().to(Params.device)


# mlp = partial(RFFMLP, 1, 10, 50, n_layers, 50)
# mlp = partial(StackedLFF, 1, 10, 50, 10, n_layers, 50)
# mlp = partial(RFFMLP, 1, B_scale, 50, 1, 50)
def train_sine(charts=None, **deps):
    from ml_logger import logger

    # speed up training.
    torch.set_float32_matmul_precision("medium")  # or high
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # update params and register experiment
    Params._update(deps)
    logger.log_params(Params=vars(Params))
    if charts is not False:
        # fmt: off
        logger.log_text(charts or """
        charts:
        - xKey: epoch
          yKey: loss/mean
          yDomain: [-0.01, 1.0]
        - glob: plots/*.png
          type: image
        - glob: kernels/*.png
          type: image
        keys:
        - Params.lr
        - Params.batch_size
        """, dedent=True, filename=".charts.yml", overwrite=True)
    # fmt: on
    logger.upload_file(__file__)

    # setting the seed here for reproducible results
    # random.seed(Params.seed)
    np.random.seed(Params.seed)
    torch.manual_seed(Params.seed)

    xs = np.linspace(0, 1, 801)
    xs_sparse = np.linspace(0, 1, 41)

    if Params.target == "sine-ensemble":
        sines = [np.sin(2 * np.pi * (k * xs + p)) for k, p in zip(np.arange(5, 60, 5), np.random.random(11))]
    elif Params.target == "sine":
        k = Params.target_k
        sines = [
            np.sin(2 * np.pi * k * xs),
        ]
    elif Params.target == "sine-ensemble-test":
        sines = [np.sin(2 * np.pi * (k * xs)) for k in np.arange(3, 58, 5)]
    else:
        raise RuntimeError(f"target {Params.target} is not allowed.")

    ys = np.sum(sines, axis=0)

    if Params.checkpoint is not None:
        net = logger.load_torch(Params.checkpoint, map_location=Params.device)
    elif Params.network == "mlp":
        mlp = partial(MLP, 1, Params.lat_dim, Params.n_layers, 1)
        net = mlp().to(Params.device)
    elif Params.network == "ffn":
        FFN = lambda: nn.Sequential(
            LFF(1, Params.lat_dim, scale=Params.ffn_scale), MLP(Params.lat_dim, Params.lat_dim, Params.n_layers - 1, 1)
        )
        net = FFN().to(Params.device)

    buffer = Buffer(xs, ys)

    optim = Adam(net.parameters(), Params.lr)

    for epoch in range(Params.n_epochs + 1):
        for x, y in buffer.sample(Params.batch_size):
            out = net(x[:, None].to(Params.device))
            loss = smooth_l1_loss(out, y[:, None].to(Params.device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            with logger.Prefix(metrics=Params.metrics_prefix):
                logger.store_metrics(loss=loss.item())

        if logger.every(Params.log_interval, key="epoch", start_on=1):
            logger.log_metrics_summary(key_values={"epoch": epoch})
        if epoch in Params.checkpoint_stops or epoch % Params.checkpoint_interval == 0:
            logger.torch_jit_save(net, f"checkpoints/net_{epoch}.pt")
            logger.duplicate(f"checkpoints/net_{epoch}.pt", "checkpoints/net_last.pt")

            gram = get_ntk(net, xs_sparse)
            plt.imshow(gram)
            logger.savefig(f"kernels/kernel_{epoch:08d}.png", close=True)
            # logger.save_pkl(dict(epoch=epoch, gram=gram), path=f"gram.pkl", append=True)
        if logger.every(Params.plot_interval, key="ntk", start_on=1):
            plot_fit(xs, ys, net, filename=f"plots/fit_{epoch:08}.png")


def wrap(**deps):
    from noisy_ntk.sine_mlp import train_sine

    return train_sine(**deps)


if __name__ == "__main__":
    import jaynes
    from ml_logger.job import instr, RUN

    jaynes.config()

    RUN.job_name += "-mlp/{job_counter}"
    RUN.CUDA_VISIBLE_DEVICES = "1"

    thunk = instr(wrap, lr=1e-4, batch_size=801, log_interval=1000, seed=400)
    jaynes.add(thunk)

    thunk = instr(wrap, lr=1e-4, batch_size=801, log_interval=1000, seed=500)
    jaynes.chain(thunk)

    thunk = instr(wrap, lr=1e-4, batch_size=801, log_interval=1000, seed=600)
    jaynes.chain(thunk)

    job_ids = jaynes.execute()
    # jaynes.listen(command=f"kubectl logs -f {job_ids[0]} --all-containers", interval=5)
    jaynes.listen()

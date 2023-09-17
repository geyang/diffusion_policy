from ml_logger.job import RUN, instr
from params_proto.hyper import Sweep

from noisy_neurons.pytorch_mnist import Args, main

if __name__ == "__main__":

    with Sweep(RUN, Args).product as sweep:
        Args.net = ["CNN", "MLP"]
        Args.seed = [100, 200, 300]

    for deps in sweep:
        thunk = instr(main)
        thunk(**deps)

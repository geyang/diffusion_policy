from params_proto import ParamsProto

from params_proto.hyper import Sweep


class Machines(ParamsProto):
    """
    # Machine Config Type

    You can import machine from this module. You can comment out
    the machines in the jsonl to disable them.
    """

    ip = "visiongpu54.csail.mit.edu"
    gpu_id = 0


try:
    machines = Sweep(Machines).load("machines.jsonl")
except FileNotFoundError:
    with Sweep(Machines).product as machines:
        Machines.ip = [
            "vision26.csail.mit.edu",
            "vision28.csail.mit.edu",
            # "visiongpu52.csail.mit.edu",
            "visiongpu53.csail.mit.edu",
            "visiongpu54.csail.mit.edu",
        ]
        Machines.gpu_id = [0, 1, 2, 3, 4, 5, 6, 7]

    machines.save("machines.jsonl")

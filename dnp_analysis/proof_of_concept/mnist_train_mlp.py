from params_proto.hyper import Sweep

from dnp.mnist_mlp import main, Params

if __name__ == "__main__":
    import jaynes
    from dnp_analysis import instr, RUN

    machines = [
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=0),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=1),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=2),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=3),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=4),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=5),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=6),
        dict(ip="visiongpu54.csail.mit.edu", gpu_id=7),
        dict(ip="visiongpu55.csail.mit.edu", gpu_id=0),
        dict(ip="visiongpu55.csail.mit.edu", gpu_id=1),
        dict(ip="visiongpu55.csail.mit.edu", gpu_id=2),
        dict(ip="visiongpu55.csail.mit.edu", gpu_id=3),
        dict(ip="visiongpu56.csail.mit.edu", gpu_id=0),
        dict(ip="visiongpu56.csail.mit.edu", gpu_id=1),
    ]

    default_job_name = RUN.job_name

    with Sweep(RUN, Params).product as sweep:

        Params.lr = [0.1, 1e-2, 5e-3, 0.001, 0.0005, 0.0001]
        Params.seed = [100, 200]

    for i, deps in sweep.items():
        if i % 2 == 0:
            ip, gpu_id = machines.pop().values()

        jaynes.config(launch={"ip": ip})

        RUN.CUDA_VISIBLE_DEVICES = f"{gpu_id}"
        RUN.job_name = f"{default_job_name}/lr-{Params.lr}/{Params.seed}"
        thunk = instr(main, **deps)

        if i % 2 == 0:
            jaynes.add(thunk)
        else:
            jaynes.chain(thunk)

    jaynes.execute()
    jaynes.listen()
    # job_ids = jaynes.execute()
    # command = f"kubectl logs -f {job_ids[0]} --all-containers"
    # print(command)
    # jaynes.listen(command=command, interval=5)

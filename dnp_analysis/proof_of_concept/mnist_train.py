from dnp.mnist_cnn import main

if __name__ == '__main__':
    import jaynes
    from dnp_analysis import instr, RUN

    jaynes.config(launch=dict(ip="visiongpu52.csail.mit.edu"))

    _ = RUN.job_name

    for i, seed in enumerate([100, 200, 300, 400, 500]):
        RUN.CUDA_VISIBLE_DEVICES = f"{i}"
        RUN.job_name = f"{_}/{seed}"
        thunk = instr(main, seed=seed, target=None)
        if i % 3 == 0:
            jaynes.add(thunk)
        else:
            jaynes.chain(thunk)

    jaynes.execute()
    jaynes.listen()
    # job_ids = jaynes.execute()
    # command = f"kubectl logs -f {job_ids[0]} --all-containers"
    # print(command)
    # jaynes.listen(command=command, interval=5)

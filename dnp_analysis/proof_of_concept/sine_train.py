from dnp.sine_mlp import train_sine

if __name__ == '__main__':
    import jaynes
    from dnp_analysis import instr, RUN

    jaynes.config()

    _ = RUN.job_name

    for i, seed in enumerate([100, 200, 300, 400, 500]):
        RUN.CUDA_VISIBLE_DEVICES = f"{i}"
        RUN.job_name = f"{_}/{seed}"
        thunk = instr(train_sine, lr=1e-4, batch_size=801, log_interval=1000, seed=seed)
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

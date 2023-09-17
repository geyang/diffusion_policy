from dnp.mnist_cnn import main

if __name__ == '__main__':
    import jaynes
    from ml_logger import instr, RUN

    jaynes.config("local")

    _ = RUN.job_name

    # for i, seed in enumerate([100, 200, 300, 400, 500][:1]):
    seed = 100
    # RUN.CUDA_VISIBLE_DEVICES = f"{i}"
    RUN.job_name = f"{_}/{seed}"
    thunk = instr(main, seed=seed, target="random", batch=800, n_epochs=200, charts="""
    charts:
    - yKeys: ["train/accuracy/mean", "test/accuracy/mean"]
      xKey: epoch
    - yKeys: ["train/loss/mean"]
      xKey: epoch
    - yKeys: ["lr/mean"]
      xKey: epoch
    """)
    jaynes.run(thunk)
    jaynes.listen()

    # jaynes.execute()
    # jaynes.listen()
    # job_ids = jaynes.execute()
    # command = f"kubectl logs -f {job_ids[0]} --all-containers"
    # print(command)
    # jaynes.listen(command=command, interval=5)

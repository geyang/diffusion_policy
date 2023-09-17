from dnp.sine_mlp import train_sine

if __name__ == '__main__':
    import jaynes
    from ml_logger import instr, RUN

    jaynes.config()

    RUN.job_name += "-finetune/{job_counter}"

    charts = """
    keys:
    - Params.lr
    - Params.batch_size
    charts:
    - xKey: epoch
      yKey: ffn-10/loss/mean
      yDomain: [-0.1, 1.0]
    - xKey: epoch
      yKey: mlp-1000k/loss/mean
      yDomain: [-0.1, 1.0]
    - xKey: epoch
      yKey: ffn-1/loss/mean
      yDomain: [-0.1, 1.0]
    - xKey: epoch
      yKey: ffn-3/loss/mean
      yDomain: [-0.1, 1.0]
    - xKey: epoch
      yKey: ffn-5/loss/mean
      yDomain: [-0.1, 1.0]
    """

    for i in range(1, 4):
        thunk = instr(train_sine, target="sine-test", lr=1e-4, batch_size=801, log_interval=1000, charts=False,
                      seed=1000, checkpoint_interval=None, kernel_interval=None)
        for epoch in range(0, 1200_000, 200_000):
            checkpoint = f"/geyang/scratch/2022/06-16/mit/noisy-ntk/dhn/sine_mlp/19.07.48-ffn/" \
                         f"{i}/checkpoints/net_{epoch}.pt"
            jaynes.add(thunk, checkpoint=checkpoint,
                       metrics_prefix=f"mlp-{epoch // 1000}k", n_epochs=400, log_interval=1, plot_interval=50,
                       charts=False)
        jaynes.chain(thunk, network="ffn", ffn_scale=1, metrics_prefix="ffn-1", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=3, metrics_prefix="ffn-3", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=5, metrics_prefix="ffn-5", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=10, metrics_prefix="ffn-10", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=20, metrics_prefix="ffn-20", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=40, metrics_prefix="ffn-40", n_epochs=400, log_interval=1,
                     plot_interval=10)
        jaynes.chain(thunk, network="ffn", ffn_scale=80, metrics_prefix="ffn-80", n_epochs=400, log_interval=1,
                     plot_interval=10, charts=charts)

    job_ids = jaynes.execute()
    jaynes.listen()
    # jaynes.listen(command=f"kubectl logs -f {job_ids[0]} --all-containers", interval=5)

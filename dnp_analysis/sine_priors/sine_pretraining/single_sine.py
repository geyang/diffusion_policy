from pathlib import Path

from dnp.sine_mlp import Params

if __name__ == "__main__":
    from dnp_analysis import RUN
    from params_proto.hyper import Sweep

    # the sweep
    with Sweep(RUN, Params) as sweep:
        Params.lr = 1e-4

        Params.target = "sine"
        Params.batch_size = 801
        Params.log_interval = 50
        Params.n_epochs = 40_000
        with sweep.product:
            Params.target_k = [5, 10, 15, 20, 25, 30, 35, 40]  # 25, 30, 35, 40, 45, 50]
            Params.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def tail(RUN: RUN, Params):
        RUN.job_name = f"k-{Params.target_k}/{Params.seed}"
        RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__)

    sweep.save(f"{Path(__file__).stem}.jsonl")

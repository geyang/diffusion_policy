from pathlib import Path

from ml_logger.job import RUN
from params_proto.hyper import Sweep

from diffusion_policy.ddpm import DDPM

with Sweep(DDPM, RUN) as sweep:
    # there is no num_steps for inference
    # DDPM.n

    with sweep.product:
        DDPM.dataset = ["m1", "m2", "m_odd", "m_even", "m_three", "m_a", "m_b", "m_c"]
        DDPM.seed = [100, 200, 300]


def tail(DDPM, RUN):
    # RUN.prefix = f"pql/toy_diffusion/{}/{}/ddpm/base/{DDPM.dataset}/{DDPM.seed}"
    RUN.job_name = f"ddpm/base/{DDPM.dataset}/{DDPM.seed}"


sweep.each(tail).save(f"{Path(__file__).stem}.jsonl")

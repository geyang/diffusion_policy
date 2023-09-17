from pathlib import Path

from ml_logger.job import instr, RUN

RUN.project = "denoising-prior"
# RUN.prefix = "{project}/{project}/{username}/{now:%Y/%m-%d}/{file_stem}/{job_name}"
RUN.prefix = "{project}/{project}/{file_stem}/{now:%Y-%m-%d/%H.%M.%S}/{job_name}"
RUN.job_counter = ""
RUN.script_root = Path(__file__).parent
assert (instr, RUN)

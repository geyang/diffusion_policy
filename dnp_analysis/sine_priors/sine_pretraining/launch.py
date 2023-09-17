from params_proto import ParamsProto

from dnp.sine_mlp import train_sine, Params

if __name__ == "__main__":
    import jaynes
    from dnp_analysis import instr, RUN
    from params_proto.hyper import Sweep
    from dnp_analysis.infra.machines import machines

    sweep = Sweep(RUN, Params).load("single_sine.jsonl")

    machine_list = machines.list

    for i, deps in sweep.items():

        m = machine_list[i % 8]

        thunk = instr(train_sine, **deps)
        jaynes.config(
            launch=dict(ip=m["ip"]),
            runner=dict(gpus=f"'\"device={m['gpu_id']}\"'"),
        )

        jaynes.add(thunk)
        job_ids = jaynes.execute()

    jaynes.listen()

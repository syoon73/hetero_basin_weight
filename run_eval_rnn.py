import sys
from subprocess import run
from typing import List


### Option ###
gpu_id = int(sys.argv[1])
start_seed = int(sys.argv[2])
end_seed = int(sys.argv[3])

len_train = int(sys.argv[4])
length = 270

epoch = int(sys.argv[5])
last_epoch = False
clip = False

do_ensemble = [False, True][int(sys.argv[6])]
result_dir = f"../log{epoch}{'_last_epoch' if last_epoch else ''}{'_clip' if clip else ''}/{len_train}yrs"

reduce_static = [False, True][int(sys.argv[7])]
augment_de = [False, True][int(sys.argv[8])]

python_str = "python3"

args = {
    "gpu_id": gpu_id,
    "data_dir_us": "/media/syoon/hydro/data/SemiLSTM/Kratzert",
    # "data_dir_de": "/media/syoon/hydro/data/SemiLSTM/Germany_Data",
    "len_train": len_train,
    "reduce_static": reduce_static,
    "augment_de": augment_de,
    "length": length,
    "shift": 0,
    "rnn_type": "lstm",
    "time_embed": False,
    "hidden_size": 256,
    "num_rnn_layers": 1,
    "num_regression_layers": 1,
    "input_dropout": 0,  # 0.4
    "rnn_dropout": 0,
    "regression_dropout": 0,  # 0.4
    "sl_mode": "ef",
    "lam": 0.25,
    "mu": 0.0,
    "discard_negative_similarity": True,
    # "num_parallel_groups": 2,
    # "num_basin_batches": 1,
    "batch_size": 256,
    "num_workers": 1,
    # "lr": 1e-03,
    "weight_decay": 1e-05,
    "epoch": epoch,
}
##############


def valid_per_seed(seed: int) -> None:
    script_name = f"valid_rnn.py"
    args_str = " ".join(
        f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
        for k, v in args.items()
        if not (isinstance(v, bool) and (not v))
    )

    cmd = f"{python_str} {script_name} {args_str} --seed {seed}"

    for additional_args_str in [
        "",
    ]:
        run(
            f"{cmd} {additional_args_str}", cwd=f"./hetero_basin_weight", shell=True
        )


def ensemble(
    seed_list: List[int],
    last_epoch: bool = False,
    clip: bool = False,
    result_dir: str = "./log",
) -> None:
    script_name = f"ensemble_rnn.py"
    args_str = " ".join(
        f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
        for k, v in args.items()
        if not (isinstance(v, bool) and (not v))
    )
    seed_list_str = f'--seed_list {" ".join([str(_) for _ in seed_list])}'

    cmd = f'{python_str} {script_name} {args_str}{" --last_epoch" if last_epoch else ""}{" --clip" if clip else ""} --result_dir {result_dir}'

    for additional_args_str in [""]:
        run(
            f"{cmd} {additional_args_str} {seed_list_str}",
            cwd=f"./hetero_basin_weight",
            shell=True,
        )


def main() -> None:
    seed_list = list(range(start_seed, end_seed + 1))

    if not last_epoch:
        for seed in seed_list:
            valid_per_seed(seed)

    if do_ensemble:
        ensemble(seed_list, last_epoch=last_epoch, clip=clip, result_dir=result_dir)


if __name__ == "__main__":
    main()

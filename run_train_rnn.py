import sys
from subprocess import run


### Option ###
gpu_id = int(sys.argv[1])
start_seed = int(sys.argv[2])
end_seed = int(sys.argv[3])

len_train = int(sys.argv[4])
length = 270

epoch = int(sys.argv[5])

reduce_static = [False, True][int(sys.argv[6])]
augment_de = [False, True][int(sys.argv[7])]

python_str = "python3"

args = {
	"gpu_id": gpu_id,
	"data_dir_us": "/media/syoon/hydro/data/SemiLSTM/Kratzert",
	"data_dir_de": "/media/syoon/hydro/data/SemiLSTM/Germany_Data",
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
	"sim_thr": 0,
	"sim_scale": 1.96,
	"lam": 0.25,
	"mu": 0.0,
	"discard_negative_similarity": True,
	"num_basin_batches": 1,
	"batch_size": 256,
	"num_workers": 1,
	"lr": 1e-03,
	"weight_decay": 1e-05,
	"epoch": epoch,
}
##############


def train_per_seed(seed: int) -> None:
	script_name = f"train_rnn.py"
	args_str = " ".join(
		f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
		for k, v in args.items()
		if not (isinstance(v, bool) and (not v))
	)

	cmd = f"{python_str} {script_name} {args_str} --seed {seed}"

	for additional_args_str in [
		"",
	]:
		run(f"{cmd} {additional_args_str}", cwd=f"./hetero_basin_weight", shell=True)


def main() -> None:
	seed_list = list(range(start_seed, end_seed + 1))

	for seed in seed_list:
		train_per_seed(seed)


if __name__ == "__main__":
	main()

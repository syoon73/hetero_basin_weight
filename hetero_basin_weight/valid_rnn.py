import argparse
import numpy
import sys
import time
import torch
from os.path import abspath, dirname, exists
from typing import Tuple

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)
from lib.data import (
	X_DIM,
	Z_DIM,
	Z_DIM_REDUCED,
	HydrologyDatasetSegmentation,
	prepare_data,
)
from lib.metric import kge_loss, nse_loss
from lib.model import RNN
from lib.nnfn import inference_rnn
from lib.util import (
	get_argparser,
	get_args_str,
	get_basin_list,
	get_start_epoch,
	mean_std_unnorm,
)


def run_target(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
	target_basin: str,
	data_dev: Tuple[numpy.ndarray, numpy.ndarray],
	ystat: Tuple[numpy.ndarray, numpy.ndarray],
) -> None:
	# Model directory
	model_dir = abspath(
		f"./model{args.len_train}/{model_name}/{target_basin}/seed{args.seed}"
	)
	if not exists(model_dir):
		print(f"{model_dir} does not exist. Skip.")
		return

	# Log path
	log_path = f"{model_dir}/log.txt"
	start_epoch = get_start_epoch(log_path)

	if start_epoch > args.epoch:
		print(f"The validation for {model_dir} has already done. Skip.")
		return

	#
	x_dev, y_dev = data_dev
	y_mean, y_std = ystat

	loader_dev = torch.utils.data.DataLoader(
		HydrologyDatasetSegmentation(
			x_dev, y_dev, args.length, stride=1, basin_idx=1108, skip_nan_y=True	# basin_idx has no meaning in evaluation, so put an arbitrary value
		),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)

	# Validation
	while start_epoch <= args.epoch:
		network_path = f"{model_dir}/{start_epoch}"

		if not exists(network_path):
			sys.stdout.write(f"\r{network_path} does not exists. Wait...\n")
			sys.stdout.flush()
			time.sleep(args.sleep)
			continue

		# Load network
		network = RNN(
			input_size=X_DIM + (Z_DIM_REDUCED if args.reduce_static else Z_DIM),
			output_size=1,
			**vars(args),
		)
		network.load_state_dict(torch.load(network_path))
		network.to(device)

		# Compute
		pred_dev, target_dev = inference_rnn(network, loader_dev, device)

		# Restore to the original scale
		pred_dev = mean_std_unnorm(pred_dev, y_mean, y_std)

		# (1,) --> (,)
		nse_dev = nse_loss(pred_dev, target_dev).squeeze()
		kge_dev = kge_loss(pred_dev, target_dev).squeeze()

		# Write log
		with open(f"{model_dir}/log.txt", "a") as f:
			f.write(f"{start_epoch}\t{nse_dev}\t{kge_dev}\n")

		print(
			f"{start_epoch}\tdev nse:\t{nse_dev:.4f}\tdev kge:\t{kge_dev:.4f}\t{log_path}"
		)

		start_epoch += 1

	print(f"Done {log_path}")


def run(args: argparse.Namespace, model_name: str, device: torch.device) -> None:
	# Basins
	basin_list = sorted(get_basin_list(f"{args.data_dir_us}/global.txt"))
	# basin_list = get_basin_list(f"{args.data_dir_us}/global_shuffled.txt")

	# Load data
	basin2data_dev = {}
	basin2ystat = {}

	for basin in basin_list:
		(
			_,
			(x_dev_b, y_dev_b),
			_,
			(y_mean_b, y_std_b),
		) = prepare_data(
			basin,
			args.data_dir_us,
			args.len_train,
			args.length,
			shift=args.shift,
			normalize_y=False,
			time_embed=args.time_embed,
			drop_front_train=False,
			is_de=False,
			reduce_static=args.reduce_static,
		)
		basin2data_dev[basin] = (x_dev_b, y_dev_b)
		basin2ystat[basin] = (y_mean_b, y_std_b)

	for target_basin in basin_list:
		run_target(
			args,
			model_name,
			device,
			target_basin,
			basin2data_dev[target_basin],
			basin2ystat[target_basin],
		)


def main() -> None:
	# Arguments
	parser = get_argparser()
	args = parser.parse_args()

	# Setting
	numpy.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	assert torch.cuda.is_available()
	torch.backends.cudnn.deterministic = True
	device = torch.device(f"cuda:{args.gpu_id}")

	# Workspace
	model_name = get_args_str(args)

	#
	run(args, model_name, device)


if __name__ == "__main__":
	main()

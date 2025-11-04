import argparse
import numpy
import sys
import torch
from os import makedirs
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
from lib.nnfn import inference_rnn
from lib.metric import kge_loss, nse_loss
from lib.model import RNN
from lib.util import (
	get_argparser,
	get_args_str,
	get_basin_list,
	get_best_epoch,
	mean_std_unnorm,
)


def run_target(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
	target_basin: str,
	data_dev: Tuple[numpy.ndarray, numpy.ndarray],
	data_eval: Tuple[numpy.ndarray, numpy.ndarray],
	ystat: Tuple[numpy.ndarray, numpy.ndarray],
) -> Tuple[Tuple[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]]:
	#
	x_dev, y_dev = data_dev
	x_eval, y_eval = data_eval
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
	loader_eval = torch.utils.data.DataLoader(
		HydrologyDatasetSegmentation(
			x_eval, y_eval, args.length, stride=1, basin_idx=1108, skip_nan_y=True	# basin_idx has no meaning in evaluation, so put an arbitrary value
		),
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)

	#
	pred_dev = []
	pred_eval = []

	target_dev = None
	target_eval = None

	for seed in args.seed_list:
		#
		model_dir = f"./model{args.len_train}/{model_name}/{target_basin}/seed{seed}"
		assert exists(model_dir)

		# Model selection
		if args.last_epoch:
			best_epoch = args.epoch
		else:
			log_path = f"{model_dir}/log.txt"
			best_epoch = get_best_epoch(
				log_path, key_order=[1, 0], max_epoch=args.epoch
			)
		network_path = f"{model_dir}/{best_epoch}"

		# Load network
		network = RNN(
			input_size=X_DIM + (Z_DIM_REDUCED if args.reduce_static else Z_DIM),
			output_size=1,
			**vars(args),
		)
		network.load_state_dict(torch.load(network_path))
		network.to(device)

		# Compute
		pred_dev_s, target_dev_s = inference_rnn(network, loader_dev, device)
		pred_eval_s, target_eval_s = inference_rnn(network, loader_eval, device)

		pred_dev.append(pred_dev_s)
		pred_eval.append(pred_eval_s)

		if target_dev is None:
			target_dev = target_dev_s
		else:
			assert numpy.array_equal(target_dev, target_dev_s)

		if target_eval is None:
			target_eval = target_eval_s
		else:
			assert numpy.array_equal(target_eval, target_eval_s)

	# (# seeds + 1, len, C=1)
	pred_dev = numpy.stack(pred_dev)
	pred_dev = numpy.vstack([pred_dev, numpy.nanmean(pred_dev, axis=0, keepdims=True)])
	pred_eval = numpy.stack(pred_eval)
	pred_eval = numpy.vstack(
		[pred_eval, numpy.nanmean(pred_eval, axis=0, keepdims=True)]
	)

	# (# seeds + 1, len, C=1) --> (# seeds + 1, len)
	# Restore to the original scale
	pred_dev = mean_std_unnorm(pred_dev, y_mean, y_std).squeeze(-1)
	pred_eval = mean_std_unnorm(pred_eval, y_mean, y_std).squeeze(-1)

	if args.clip:
		pred_dev = numpy.maximum(pred_dev, 0)
		pred_eval = numpy.maximum(pred_eval, 0)

	# (len, C=1) --> (1, len, C=1) --> (1, len)
	target_dev = numpy.expand_dims(target_dev, 0).squeeze(-1)
	target_eval = numpy.expand_dims(target_eval, 0).squeeze(-1)

	assert (
		pred_dev.shape[0] == pred_eval.shape[0] == len(args.seed_list) + 1
		and pred_dev.shape[1] == target_dev.shape[1]
		and pred_eval.shape[1] == target_eval.shape[1]
	)

	return (pred_dev, target_dev), (pred_eval, target_eval)


def run(args: argparse.Namespace, model_name: str, device: torch.device) -> None:
	# Result (i.e., ensembled log) path
	result_path = f"{args.result_dir}/{model_name}.txt"
	if exists(result_path):
		print(f"{result_path} already exists. Skip.")
		return

	# Basins
	basin_list = sorted(get_basin_list(f"{args.data_dir_us}/global.txt"))
	# basin_list = get_basin_list(f"{args.data_dir_us}/global_shuffled.txt")

	# Load data
	basin2data_dev = {}
	basin2data_eval = {}
	basin2ystat = {}

	for basin in basin_list:
		(
			_,
			(x_dev_b, y_dev_b),
			(x_eval_b, y_eval_b),
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
		basin2data_eval[basin] = (x_eval_b, y_eval_b)
		basin2ystat[basin] = (
			numpy.expand_dims(y_mean_b, 0),
			numpy.expand_dims(y_std_b, 0),
		)

	# (# basins, # seeds + 1, 4: {nse_dev}, {kge_dev}, {nse_eval}, {kge_eval})
	result = numpy.empty(
		(len(basin_list), len(args.seed_list) + 1, 4), dtype=numpy.float32
	)

	for b, target_basin in enumerate(basin_list):
		(pred_dev_b, target_dev_b), (pred_eval_b, target_eval_b) = run_target(
			args,
			model_name,
			device,
			target_basin,
			basin2data_dev[target_basin],
			basin2data_eval[target_basin],
			basin2ystat[target_basin],
		)

		nse_dev_b = nse_loss(pred_dev_b, target_dev_b, axis=1)
		kge_dev_b = kge_loss(pred_dev_b, target_dev_b, axis=1)

		nse_eval_b = nse_loss(pred_eval_b, target_eval_b, axis=1)
		kge_eval_b = kge_loss(pred_eval_b, target_eval_b, axis=1)

		result[b, :, 0] = nse_dev_b
		result[b, :, 1] = kge_dev_b
		result[b, :, 2] = nse_eval_b
		result[b, :, 3] = kge_eval_b

	result = result.reshape((len(basin_list), -1))

	#
	if not exists(args.result_dir):
		makedirs(args.result_dir)
	with open(result_path, "w") as f:
		for b, basin in enumerate(basin_list):
			result_b_str = "\t".join([str(f"{v:.4f}") for v in result[b]])
			f.write(f"{basin}\t{result_b_str}\n")


def main() -> None:
	# Arguments
	parser = get_argparser()
	parser.add_argument("--result_dir", type=str, required=True)
	parser.add_argument("--seed_list", type=int, nargs="+", default=list(range(0, 5)))
	parser.add_argument("--last_epoch", action="store_true", default=False)
	parser.add_argument("--clip", action="store_true", default=False)
	args = parser.parse_args()

	# Setting
	assert torch.cuda.is_available()
	torch.backends.cudnn.deterministic = True
	device = torch.device(f"cuda:{args.gpu_id}")

	# Workspace
	model_name = get_args_str(args)

	#
	run(args, model_name, device)


if __name__ == "__main__":
	main()

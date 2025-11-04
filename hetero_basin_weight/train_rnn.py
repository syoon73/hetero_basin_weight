import argparse
import numpy
import sys
import torch
from os import makedirs
from os.path import abspath, dirname, exists
from typing import List

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)
from lib.data import (
	X_DIM,
	Z_DIM,
	Z_DIM_REDUCED,
	HydrologyDatasetSegmentation,
	get_dataloader,
	get_dataloader_per_basin,
	prepare_data,
)
from lib.model import RNN
from lib.nnfn import BasinAwareSuperLoss, train_rnn_basin_aware_superloss
from lib.util import get_argparser, get_args_str, get_basin_list, save_model


def run_target(
	args: argparse.Namespace,
	model_name: str,
	device: torch.device,
	target_idx: int,
	basin_list: List[str],
	basin2data_train: dict,
	init_params: dict,
) -> None:
	# Model directory
	assert "DE" not in basin_list[target_idx]
	model_dir = abspath(
		f"./model{args.len_train}/{model_name}/{basin_list[target_idx]}/seed{args.seed}"
	)
	if exists(model_dir):
		print(f"{model_dir} already exists. Skip.")
		return

	# DataLoader
	loader_train, _ = get_dataloader(
		basin2data_train,
		HydrologyDatasetSegmentation,
		args.length,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)

	# For computing the gradients
	loader_train_per_basin = get_dataloader_per_basin(
		basin2data_train,
		HydrologyDatasetSegmentation,
		args.length,
		batch_size=args.batch_size,
		shuffle=True,  # Set `True` for approximation
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
	)
	assert len(loader_train_per_basin) == len(basin_list)

	#
	assert not exists(model_dir)
	makedirs(model_dir)
	print(f"Start: {model_dir}")

	# Create network
	network = RNN(
		input_size=X_DIM + (Z_DIM_REDUCED if args.reduce_static else Z_DIM),
		output_size=1,
		**vars(args),
	)
	network.load_state_dict(init_params)
	network.to(device)

	# Optimizer
	optimizer = torch.optim.RAdam(
		network.parameters(), lr=args.lr, weight_decay=args.weight_decay
	)

	# SuperLoss
	sl_class = {"ef": BasinAwareSuperLoss}
	sl = sl_class[args.sl_mode](
		target_idx,
		len(basin_list),
		args.lam,
		args.mu,
	)
	sl.to(device)

	# Run
	for i in range(1, args.epoch + 1):
		if i == 1:
			# Trick
			ref_path = abspath(
				f"./model{args.len_train}/{model_name}/{basin_list[0]}/seed{args.seed}/1"
			)
			if exists(ref_path):
				network.load_state_dict(torch.load(ref_path))
				print(f"Load {ref_path} as a trick.")
			else:
				train_rnn_basin_aware_superloss(
					network, loader_train, optimizer, sl, i, device
				)
		else:
			train_rnn_basin_aware_superloss(
				network, loader_train, optimizer, sl, i, device
			)

		if i < args.epoch:
			grad_sim = sl.compute_sigma(
				network,
				loader_train_per_basin,
				device,
				num_parallel_groups=args.num_parallel_groups,
				num_basin_batches=args.num_basin_batches,
			)
			print(
				f"\tgrad_sim {grad_sim.mean():.3f} {grad_sim.min():.3f} {grad_sim.max():.3f}"
			)
			numpy.save(f"{model_dir}/gs_{i}", grad_sim)
			# numpy.save(f"{model_dir}/tau_{i}", sl.tau.cpu().numpy())
			numpy.save(f"{model_dir}/w_{i}", sl.sigma.cpu().numpy())

			if args.discard_negative_similarity:
				mask = grad_sim < 0.0
				sl.sigma[mask] = 0.0

				loader_train, _ = get_dataloader(
					{
						k: v
						for b, (k, v) in enumerate(basin2data_train.items())
						if not mask[b]
					},
					HydrologyDatasetSegmentation,
					args.length,
					batch_size=args.batch_size,
					shuffle=True,
					num_workers=args.num_workers,
					pin_memory=True,
					drop_last=False,
				)

				print(f"Discarded basins: {mask.sum()} / {len(mask)}")

		else:
			print()

		# Save model
		save_model(f"{model_dir}/{i}", network)

	print(f"Done {abspath(model_dir)}")


def run(args: argparse.Namespace, model_name: str, device: torch.device) -> None:
	# Basins
	basin_list = sorted(get_basin_list(f"{args.data_dir_us}/global.txt"))
	# basin_list = get_basin_list(f"{args.data_dir_us}/global_shuffled.txt")

	# Load data
	basin2data_train = {}

	for basin in basin_list:
		(
			(x_train_b, y_train_b),
			_,
			_,
			_,
		) = prepare_data(
			basin,
			args.data_dir_us,
			args.len_train,
			args.length,
			shift=args.shift,
			time_embed=args.time_embed,
			drop_front_train=False,
			is_de=False,
			reduce_static=args.reduce_static,
		)
		basin2data_train[basin] = (x_train_b, y_train_b)

	if args.augment_de:
		basin_list_de = sorted(
			get_basin_list(f"{args.data_dir_de}/1147_gauge_list.txt")
		)
		for basin in basin_list_de:
			(
				(x_train_b, y_train_b),
				_,
				_,
				_,
			) = prepare_data(
				basin,
				args.data_dir_de,
				args.len_train,
				args.length,
				shift=args.shift,
				time_embed=args.time_embed,
				drop_front_train=False,
				is_de=True,
				reduce_static=args.reduce_static,
			)
			assert basin not in basin2data_train
			basin2data_train[basin] = (x_train_b, y_train_b)
		basin_list += basin_list_de

	#
	network = RNN(
		input_size=X_DIM + (Z_DIM_REDUCED if args.reduce_static else Z_DIM),
		output_size=1,
		**vars(args),
	)
	init_params = network.state_dict()

	#
	for target_idx in range(531):  # len(basin_list)):
		run_target(
			args,
			model_name,
			device,
			target_idx,
			basin_list,
			basin2data_train,
			init_params,
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

import argparse
import numpy
import time
import torch
from os import rename
from os.path import exists
from typing import Callable, List, Optional, Tuple, Union


def get_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()

	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--gpu_id", type=int, default=0)

	parser.add_argument("--data_dir_us", type=str, required=True)
	parser.add_argument("--data_dir_de", type=str, default="./")
	parser.add_argument("--len_train", type=int, required=True)

	parser.add_argument("--reduce_static", action="store_true", default=False)
	parser.add_argument("--augment_de", action="store_true", default=False)

	parser.add_argument("--length", type=int, default=270)
	parser.add_argument("--shift", type=int, default=0)

	parser.add_argument("--rnn_type", type=str, default="lstm")
	parser.add_argument("--time_embed", action="store_true", default=False)
	parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--num_rnn_layers", type=int, default=1)
	parser.add_argument("--num_regression_layers", type=int, default=1)
	parser.add_argument("--input_dropout", type=float, default=0)
	parser.add_argument("--rnn_dropout", type=float, default=0)
	parser.add_argument("--regression_dropout", type=float, default=0)

	parser.add_argument("--sl_mode", type=str, default="ef")
	parser.add_argument("--lam", type=float, default=0.25)
	parser.add_argument("--mu", type=float, default=0.0)
	parser.add_argument(
		"--discard_negative_similarity", action="store_true", default=False
	)
	parser.add_argument("--num_parallel_groups", type=int, default=4)
	parser.add_argument("--num_basin_batches", type=int, default=0)

	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--num_workers", type=int, default=1)

	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=0)
	parser.add_argument("--epoch", type=int, default=50)
	parser.add_argument("--sleep", type=int, default=30)

	parser.add_argument("--tag", type=str, default="")

	return parser


def get_args_str(args: argparse.Namespace) -> str:
	model_name = f"{args.rnn_type}_{'reduced_' if args.reduce_static else ''}static{'_aug' if args.augment_de else ''}_{args.length}_{args.shift}_hidden_size_{args.hidden_size}_num_layers_{args.num_rnn_layers}_{args.num_regression_layers}_basl2_{args.sl_mode}{'_dns' if args.discard_negative_similarity else ''}_{args.lam}_{args.mu}"

	if args.augment_de:
		assert args.reduce_static

	if args.time_embed:
		model_name += "_time_embed"
	if 0 < args.input_dropout < 1:
		model_name += f"_input_dropout_{args.input_dropout}"
	if 0 < args.rnn_dropout < 1:
		model_name += f"_rnn_dropout_{args.rnn_dropout}"
	if 0 < args.regression_dropout < 1:
		model_name += f"_regression_dropout_{args.regression_dropout}"
	if args.num_basin_batches > 0:
		model_name += f"_nbb_{args.num_basin_batches}"
	if args.weight_decay > 0:
		model_name += f"_weight_decay_{args.weight_decay}"
	if args.tag != "":
		model_name += f"_{args.tag}"

	return model_name


def get_basin_list(path: str) -> List[str]:
	basin_list = []
	with open(path, "rb") as f:
		for line in f:
			basin = line.decode().split()[0]
			basin_list.append(basin)
	return basin_list


def compute_mean_std(
	input: numpy.ndarray, axis: Optional[Union[int, Tuple[int]]] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	mean = numpy.nanmean(input, axis=axis, keepdims=True)
	std = numpy.nanstd(input, axis=axis, keepdims=True)
	return mean, std


def mean_std_norm(
	input: numpy.ndarray, mean: numpy.ndarray, std: numpy.ndarray
) -> numpy.ndarray:
	assert input.ndim == mean.ndim == std.ndim
	output = (input - mean) / std
	return output


def mean_std_unnorm(
	input: numpy.ndarray, mean: numpy.ndarray, std: numpy.ndarray
) -> numpy.ndarray:
	assert input.ndim == mean.ndim == std.ndim
	output = input * std + mean
	return output


def get_subsequence(
	x: numpy.ndarray,
	y: numpy.ndarray,
	subsequence_range_str: Tuple[str, str],
	length: int,
	shift: int = 0,
	drop_front: bool = False,
	total_range_str: Tuple[str, str] = ["1980-01-01", "2008-12-31"],
) -> Tuple[numpy.ndarray, numpy.ndarray]:

	total_range = [numpy.datetime64(date_str) for date_str in total_range_str]
	assert (
		shift >= 0
		and x.shape[0]
		== y.shape[0]
		== (total_range[1] - total_range[0]).item().days + 1
	)

	subsequence_range = [
		numpy.datetime64(date_str) for date_str in subsequence_range_str
	]
	assert (
		total_range[0] <= subsequence_range[0] < subsequence_range[1] <= total_range[1]
	)

	n_samples = (subsequence_range[1] - subsequence_range[0]).item().days + 1
	assert n_samples <= x.shape[0]

	i = (subsequence_range[0] - total_range[0]).item().days

	if drop_front:
		assert i >= shift
		x_sub = x[i - shift : i + n_samples - shift]
		y_sub = y[i : i + n_samples]
	else:
		assert i - length + 1 >= shift
		x_sub = x[i - length + 1 - shift : i + n_samples - shift]
		# the subsequence y[i-length+1:i] actually does not used
		y_sub = y[i - length + 1 : i + n_samples]

	assert x_sub.shape[0] == y_sub.shape[0] and not numpy.isnan(x_sub).any()

	return x_sub, y_sub


def save_model(
	model_path: str, network: Union[torch.nn.Module, torch.nn.DataParallel]
) -> None:
	if exists(model_path):
		timestamp_str = time.strftime("%Y%m%d_%H%M%S")
		print(f" Already exists: {model_path}. Move it to {model_path}_{timestamp_str}")
		rename(model_path, f"{model_path}_{timestamp_str}")

	if isinstance(network, torch.nn.DataParallel):
		torch.save(network.module.state_dict(), model_path)
	else:
		torch.save(network.state_dict(), model_path)


def read_log(path: str) -> numpy.ndarray:
	log = []

	with open(path, "rb") as f:
		for i, line in enumerate(f):
			args = line.decode().split()
			epoch = args[0]
			assert int(epoch) == i + 1
			values = [float(_) for _ in args[1:]]
			log.append(values)

	log = numpy.asarray(log, dtype=numpy.float32)

	return log


def get_best_epoch(
	log_path: str, key_order: List[int] = [2, 0, 3, 1], max_epoch: int = 100
) -> int:
	# [nse mean, nse median, kge mean, kge median]
	# nse median -> kge median -> nse mean -> kge mean
	log = read_log(log_path)
	r, c = log.shape
	assert r >= max_epoch and c == len(key_order)
	best_idx = numpy.lexsort(tuple([log[:max_epoch, i] for i in key_order]))[-1]

	return best_idx + 1


def get_start_epoch(log_path: str) -> int:
	if not exists(log_path):
		return 1
	else:
		return read_log(log_path).shape[0] + 1


def mean_std_unnorm_with_cumsum_len(
	basin_list: List[str],
	pred: numpy.ndarray,
	cumsum_len: numpy.ndarray,
	basin2ystat: dict,
) -> numpy.ndarray:
	assert (
		pred.ndim in {2, 3}
		and cumsum_len.ndim == 1
		and pred.shape[-2] == cumsum_len[-1]
	)

	_pred = pred
	for b, basin in enumerate(basin_list):
		if b == 0:
			start = 0
		else:
			start = cumsum_len[b - 1]
		end = cumsum_len[b]

		y_mean_b, y_std_b = basin2ystat[basin]

		if pred.ndim == 2:
			# pred: (T, C=1)
			_pred[start:end] = mean_std_unnorm(_pred[start:end], y_mean_b, y_std_b)
		else:
			# pred: (S, T, C=1)
			_pred[:, start:end] = mean_std_unnorm(
				_pred[:, start:end], y_mean_b, y_std_b
			)

	return _pred


def compute_metric_with_cumsum_len(
	metric_fn: Callable[..., numpy.ndarray],
	basin_list: List[str],
	pred: numpy.ndarray,
	target: numpy.ndarray,
	cumsum_len: numpy.ndarray,
	axis: int = 0,
	eps: float = 1e-02,
) -> numpy.ndarray:
	assert (
		pred.shape == target.shape
		and pred.ndim == 2
		and pred.shape[0] == cumsum_len[-1]
	)
	values = []

	for b, basin in enumerate(basin_list):
		if b == 0:
			start = 0
		else:
			start = cumsum_len[b - 1]
		end = cumsum_len[b]

		pred_b = pred[start:end]
		tar_b = target[start:end]

		value_b = metric_fn(pred_b, tar_b, axis=axis, eps=eps)

		values.append(value_b)

	values = numpy.vstack(values, dtype=numpy.float32)
	assert values.shape == (len(basin_list), 1)
	return values


def split_pred_target_by_basin(
	basin_list: List[str],
	cumsum_len: numpy.ndarray,
	pred: numpy.ndarray,
	target: numpy.ndarray,
) -> Tuple[dict, dict]:
	assert len(basin_list) == len(cumsum_len) and pred.shape[1] == target.shape[1]

	basin2pred = {}
	basin2target = {}

	for b, basin in enumerate(basin_list):
		start = 0 if b == 0 else cumsum_len[b - 1]
		end = cumsum_len[b]

		basin2pred[basin] = pred[:, start:end]
		basin2target[basin] = target[:, start:end]

	return basin2pred, basin2target


def write_csv(path: str, pred: numpy.ndarray, target: numpy.ndarray) -> None:
	# pred: (S_E, T)
	# target: (1, T)

	date_range = [
		numpy.datetime64(date_str) for date_str in ["1995-10-01", "2008-09-30"]
	]  # dev + eval

	S_E, T = pred.shape
	assert (date_range[1] - date_range[0]).item().days + 1 == T and target.shape == (
		1,
		T,
	)

	with open(path, "w") as f:
		f.write("date,")
		for s in range(S_E - 1):
			f.write(f"seed{s},")
		f.write("mean,target\n")

		for t in range(T):
			date_t = date_range[0] + t
			assert date_t <= date_range[1]
			pred_t = pred[:, t]
			target_t = target[0, t]

			f.write(f"{str(date_t)},")

			for s in range(S_E):
				f.write(f"{pred_t[s]},")

			f.write(f"{target_t}\n")

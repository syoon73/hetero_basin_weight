import datetime
import numpy
import torch
from typing import List, Tuple, Union
from .util import compute_mean_std, mean_std_norm, get_subsequence

# When only using CAMELS-US for training
# X_DIM = 5
# Z_DIM = 27

# When using but CAMES-US and -DE for training
X_DIM = 3
Z_DIM = 10
Z_DIM_REDUCED = 3


class HydrologyDatasetSegmentation(torch.utils.data.Dataset):
	def __init__(
		self,
		feats: numpy.ndarray,
		labels: numpy.ndarray,
		length: int,
		stride: int,
		basin_idx: int,
		skip_nan_y: bool = True,
	) -> None:
		self.length = int(length)
		self.stride = int(stride)

		self.feats = numpy.ascontiguousarray(feats, dtype=numpy.float32)
		self.labels = numpy.ascontiguousarray(labels, dtype=numpy.float32)
		self.basin_idx = int(basin_idx)

		T = self.feats.shape[0]
		assert T == labels.shape[0] and T > self.length

		indices = self.sliding_window_idx(T, self.length, self.stride)
		self.indices = numpy.asarray(
			[
				[start, end]
				for start, end in indices
				if not (numpy.isnan(self.labels[end - 1]) and skip_nan_y)
			],
			dtype=numpy.int32,
		)

		if len(self.indices) == 0:
			print("Warning: No data in this dataset.")

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, index) -> Tuple[numpy.ndarray, numpy.float32, int]:
		start, end = self.indices[index]
		feat = numpy.ascontiguousarray(self.feats[start:end], dtype=numpy.float32)
		label = numpy.ascontiguousarray(self.labels[end - 1], dtype=numpy.float32)
		return feat, label, self.basin_idx

	@staticmethod
	def sliding_window_idx(T: int, M: int, L: int) -> numpy.ndarray:
		assert T > M
		n_remainders = (T - M) % L
		start_indices = numpy.arange(0, T - M + L - n_remainders, L, dtype=numpy.int32)
		end_indices = start_indices + M
		indices = numpy.vstack([start_indices, end_indices]).T

		if n_remainders == 0:
			assert T == indices[-1, -1]
		else:
			assert 0 < n_remainders < L
			indices = numpy.vstack([indices, [T - M, T]])

		return indices


def prepare_data(
	basin: str,
	data_dir: str,
	len_train: int,
	seqlen: int,
	shift: int = 0,
	normalize_y: bool = True,
	time_embed: bool = False,
	drop_front_train: bool = False,
	is_de: bool = False,
	reduce_static: bool = False,
) -> Tuple[
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
	Tuple[numpy.ndarray, numpy.ndarray],
]:
	if is_de:
		# DE: ["1951-01-01", "2020-12-31"]
		total_range_str: Tuple[str, str] = ["1951-01-01", "2020-12-31"]
	else:
		# US: ["1980-01-01", "2008-12-31"]
		total_range_str: Tuple[str, str] = ["1980-01-01", "2008-12-31"]

	# Data
	x = numpy.load(f"{data_dir}/forcing/{basin}.npy")  # (T, D1)
	if not is_de:
		# US: 5 --> 3
		x = x[:, [0, 2, 3]]

	assert x.shape[1] == X_DIM
	y_name = "discharge" if is_de else "stream"
	y = numpy.load(f"{data_dir}/{y_name}/{basin}.npy")  # (T, )
	y = numpy.expand_dims(y, 1)  # (T, ) -> (T, 1)

	total_range = [
		datetime.datetime.strptime(date_str, "%Y-%m-%d") for date_str in total_range_str
	]
	assert x.shape[0] == y.shape[0] == (total_range[1] - total_range[0]).days + 1

	# Normalize
	x_mean, x_std = compute_mean_std(x, axis=0)
	y_mean, y_std = compute_mean_std(y, axis=0)
	x = mean_std_norm(x, x_mean, x_std)
	if normalize_y:
		y = mean_std_norm(y, y_mean, y_std)

	# Time embedding (optional)
	if time_embed:
		t = numpy.empty((x.shape[0], 3), dtype=x.dtype)
		for i in range(x.shape[0]):
			cur_date = total_range[0] + datetime.timedelta(days=i)
			day_of_week = float(cur_date.weekday()) / 6.0 - 0.5
			day_of_month = float(cur_date.day - 1) / 31.0 - 0.5
			day_of_year = (float(cur_date.strftime("%j")) - 1) / 366.0 - 0.5
			t[i] = (day_of_week, day_of_month, day_of_year)

		assert cur_date == total_range[-1]
		x = numpy.hstack([t, x])
		assert x.shape[1] == 3 + X_DIM

	#
	assert 1 <= len_train <= 15
	train_range_str = [f"{1995-len_train}-10-01", "1995-09-30"]

	if len_train <= 2:
		dev_range_str = ["1995-10-01", "1996-09-30"]
	elif len_train == 3:
		dev_range_str = ["1995-10-01", "1997-09-30"]
	else:
		dev_range_str = ["1995-10-01", "1998-09-30"]
	eval_range_str = ["1998-10-01", "2008-09-30"]

	x_train, y_train = get_subsequence(
		x,
		y,
		subsequence_range_str=train_range_str,
		length=seqlen,
		shift=shift,
		drop_front=drop_front_train,
		total_range_str=total_range_str,
	)
	x_dev, y_dev = get_subsequence(
		x,
		y,
		subsequence_range_str=dev_range_str,
		length=seqlen,
		shift=shift,
		drop_front=False,
		total_range_str=total_range_str,
	)
	x_eval, y_eval = get_subsequence(
		x,
		y,
		subsequence_range_str=eval_range_str,
		length=seqlen,
		shift=shift,
		drop_front=False,
		total_range_str=total_range_str,
	)

	# Static
	z = numpy.load(f"{data_dir}/static/{basin}.npy")  # (D2, )
	z_mean = numpy.load(f"{data_dir}/static/mean.npy")  # (D2, )
	z_std = numpy.load(f"{data_dir}/static/std.npy")  # (D2, )

	if is_de:
		# DE: z.shape == (10+1,)
		if reduce_static:
			z = z[[2, 3, 10]]
			z_mean = z_mean[[2, 3, 10]]
			z_std = z_std[[2, 3, 10]]
		else:
			z = z[:-1]
			z_mean = z_mean[:-1]
			z_std = z_std[:-1]
	else:
		# US: z.shape == (27,)
		if reduce_static:
			loc = numpy.load(f"{data_dir}/location/{basin}.npy")  # (2, )
			loc_mean = numpy.load(f"{data_dir}/location/mean.npy")  # (2, )
			loc_std = numpy.load(f"{data_dir}/location/std.npy")  # (2, )

			z = numpy.hstack([z[[3, 5]], loc[1]], dtype=numpy.float32)  # (2 + 1,)
			z_mean = numpy.hstack(
				[z_mean[[3, 5]], loc_mean[1]], dtype=numpy.float32
			)  # (2 + 1,)
			z_std = numpy.hstack(
				[z_std[[3, 5]], loc_std[1]], dtype=numpy.float32
			)  # (2 + 1,)
		else:
			z = z[[0, 2, 3, 5, 6, 7, 8, 19, 21, 22]]
			z_mean = z_mean[[0, 2, 3, 5, 6, 7, 8, 19, 21, 22]]
			z_std = z_std[[0, 2, 3, 5, 6, 7, 8, 19, 21, 22]]

	assert (
		z.shape
		== z_mean.shape
		== z_std.shape
		== ((Z_DIM_REDUCED if reduce_static else Z_DIM),)
	)  # == (D2,)
	z = mean_std_norm(z, z_mean, z_std)

	x_train = numpy.hstack(
		[x_train, numpy.broadcast_to(z, (x_train.shape[0], z.shape[0]))]
	)
	x_dev = numpy.hstack([x_dev, numpy.broadcast_to(z, (x_dev.shape[0], z.shape[0]))])
	x_eval = numpy.hstack(
		[x_eval, numpy.broadcast_to(z, (x_eval.shape[0], z.shape[0]))]
	)

	return (x_train, y_train), (x_dev, y_dev), (x_eval, y_eval), (y_mean, y_std)


def get_dataloader(
	basin2data: dict,
	dataset_class: torch.utils.data.Dataset,
	seqlen: int,
	skip_nan_y: bool = True,
	batch_size: int = 256,
	shuffle: bool = False,
	num_workers: int = 1,
	pin_memory: bool = True,
	drop_last: bool = False,
) -> Tuple[torch.utils.data.DataLoader, numpy.ndarray]:
	dataset = []
	cumsum_len = []

	for b, (basin, (x_b, y_b)) in enumerate(basin2data.items()):
		dataset_b = dataset_class(
			x_b, y_b, seqlen, stride=1, basin_idx=b, skip_nan_y=skip_nan_y
		)
		dataset.append(dataset_b)

		cumsum_len.append(len(dataset_b))

	cumsum_len = numpy.cumsum(cumsum_len)

	loader = torch.utils.data.DataLoader(
		torch.utils.data.ConcatDataset(dataset),
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
	)
	assert (
		len(loader) > 0 and cumsum_len[-1] > 0
	), f"Insufficient sample:\tlen: {len(loader)}\t cumsum: {cumsum_len[-1]}"

	return loader, cumsum_len


def get_dataloader_per_basin(
	basin2data: dict,
	dataset_class: torch.utils.data.Dataset,
	seqlen: int,
	skip_nan_y: bool = True,
	batch_size: int = 256,
	shuffle: bool = False,
	num_workers: int = 1,
	pin_memory: bool = True,
	drop_last: bool = False,
) -> List[Union[torch.utils.data.DataLoader, None]]:
	loader = []

	for b, (basin, (x_b, y_b)) in enumerate(basin2data.items()):
		dataset_b = dataset_class(
			x_b, y_b, seqlen, stride=1, basin_idx=b, skip_nan_y=skip_nan_y
		)
		assert len(dataset_b) > 0

		loader_b = torch.utils.data.DataLoader(
			dataset_b,
			batch_size=len(dataset_b) if batch_size == 0 else batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=drop_last,
		)
		loader.append(loader_b)

	return loader

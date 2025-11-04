import numpy
import torch
from scipy.special import lambertw
from typing import Any, List, Tuple


def _aggregate_samples(
	loader: torch.utils.data.DataLoader, num_basin_batches: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
	x_b = []
	y_b = []
	for i, (x, y, _) in enumerate(loader):
		x_b.append(x)
		y_b.append(y)

		if 0 < num_basin_batches and i + 1 == num_basin_batches:
			# Approximation by taking `num_basin_batches` minibatch(es)
			# `num_basin_batches` < 1 means aggregate all the minibatches
			break

	x_b = torch.cat(x_b)
	y_b = torch.cat(y_b)

	return x_b, y_b


def _aggregate_grouped_samples(
	loader_list: List[torch.utils.data.DataLoader], num_basin_batches: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
	x_all = []
	y_all = []

	for loader in loader_list:
		x_b, y_b = _aggregate_samples(loader, num_basin_batches=num_basin_batches)
		x_all.append(x_b)
		y_all.append(y_b)

	# Raise an error when the number of samples are different across the loaders
	x_all = torch.stack(x_all)
	y_all = torch.stack(y_all)

	return x_all, y_all


def _compute_group_loss_fn(
	named_params: dict,
	named_buffers: dict,
	network: torch.nn.Module,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
) -> torch.Tensor:
	pred = torch.func.functional_call(network, (named_params, named_buffers), x_all)
	loss = torch.nn.functional.mse_loss(pred, y_all)
	return loss


def _compute_group_grad_fn(
	named_params: dict,
	named_buffers: dict,
	network: torch.nn.Module,
	x_all: torch.Tensor,
	y_all: torch.Tensor,
) -> dict:
	# argnum=0: The 0-th argument (e.g., named_params) is shared
	return torch.func.grad(_compute_group_loss_fn, argnums=0)(
		named_params, named_buffers, network, x_all, y_all
	)


def _vectorize_named_params_per_group(
	named_params_all: dict, keys: List | Tuple
) -> torch.Tensor:
	output = []
	for k in keys:
		param = named_params_all[k].flatten(1)
		output.append(param)
	return torch.cat(output, dim=-1)


class BasinAwareSuperLoss(torch.nn.Module):
	def __init__(
		self,
		target_idx: int,
		n_basins: int,
		lam: float = 1.0,
		mu: float = 0.0,
	) -> None:
		super(BasinAwareSuperLoss, self).__init__()
		assert 0 < lam and 0 <= mu

		#
		self.target_idx = int(target_idx)
		self.n_basins = int(n_basins)

		# Hparams for \sigma
		self.lam = float(lam)
		self.mu = float(mu)

		# Automatically estimated
		self.register_buffer("sigma", torch.ones(n_basins, dtype=torch.float))

	def forward(
		self, loss: torch.Tensor, basin_idx: torch.Tensor
	) -> Tuple[torch.Tensor, torch.Tensor]:
		sigma = torch.index_select(self.sigma, 0, basin_idx)
		superloss = sigma * loss
		return superloss, sigma

	def _compute_gradient_similarity_sequential(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_basin_batches: int = 0,
	) -> numpy.ndarray:
		params = tuple(network.parameters())
		grad_all = []

		for loader in loader_per_basin:
			x, y = _aggregate_samples(loader, num_basin_batches=num_basin_batches)

			x = x.to(device)
			y = y.to(device)

			pred = network(x)
			loss_b = torch.nn.functional.mse_loss(pred, y)

			# Compute basin gradient
			grad_b = torch.autograd.grad(loss_b, params)
			grad_b = torch.nn.utils.parameters_to_vector(grad_b).detach().cpu().numpy()

			grad_all.append(grad_b)

		#
		grad_all = numpy.vstack(grad_all, dtype=numpy.float32)
		grad_all /= numpy.linalg.norm(grad_all, axis=-1, keepdims=True)
		grad_sim = numpy.matmul(grad_all, grad_all[self.target_idx])
		grad_sim[self.target_idx] = 1.0

		return grad_sim

	def _compute_gradient_similarity_parallel(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_parallel_groups: int = 4,
		num_basin_batches: int = 0,
	) -> numpy.ndarray:
		#
		torch.backends.cudnn.enabled = False
		#

		named_params = dict(network.named_parameters())
		named_buffers = dict(network.named_buffers())
		param_keys = tuple(named_params.keys())

		grad_all = []

		N = len(loader_per_basin)
		Q = N // num_parallel_groups
		R = N % num_parallel_groups

		grad_fn = torch.vmap(_compute_group_grad_fn, in_dims=(None, None, None, 0, 0))

		for i in range(Q if R == 0 else Q + 1):
			loaders_i = loader_per_basin[
				i * num_parallel_groups : min((i + 1) * num_parallel_groups, N)
			]
			x_i, y_i = _aggregate_grouped_samples(
				loaders_i, num_basin_batches=num_basin_batches
			)

			grad = grad_fn(
				named_params,
				named_buffers,
				network,
				x_i.to(device),
				y_i.to(device),
			)
			grad = _vectorize_named_params_per_group(grad, param_keys).detach().cpu()

			grad_all.append(grad)

		grad_all = torch.cat(grad_all).numpy()
		grad_all /= numpy.linalg.norm(grad_all, axis=-1, keepdims=True)
		grad_sim = numpy.matmul(grad_all, grad_all[self.target_idx])
		grad_sim[self.target_idx] = 1.0

		#
		torch.backends.cudnn.enabled = True
		#

		return grad_sim

	def _compute_gradient_similarity(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_parallel_groups: int = 4,
		num_basin_batches: int = 0,
	) -> numpy.ndarray:
		#
		network.eval()
		### For `cudnn RNN backward` ###
		if hasattr(network, "rnn"):
			assert isinstance(network.rnn, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
			network.rnn.train()
		################################

		if num_parallel_groups > 1:
			return self._compute_gradient_similarity_parallel(
				network,
				loader_per_basin,
				device,
				num_parallel_groups=num_parallel_groups,
				num_basin_batches=num_basin_batches,
			)
		else:
			return self._compute_gradient_similarity_sequential(
				network, loader_per_basin, device, num_basin_batches=num_basin_batches
			)

	def compute_sigma(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_parallel_groups: int = 4,
		num_basin_batches: int = 0,
	) -> numpy.ndarray:
		#
		grad_sim = self._compute_gradient_similarity(
			network,
			loader_per_basin,
			device,
			num_parallel_groups=num_parallel_groups,
			num_basin_batches=num_basin_batches,
		)

		#
		eps = numpy.finfo(numpy.float32).eps

		beta = -grad_sim / self.lam
		z = numpy.maximum(-numpy.exp(-1) + eps, beta * 0.5)

		self.sigma = torch.from_numpy(numpy.exp(-lambertw(z).real)).to(
			dtype=torch.float, device=device
		)

		return grad_sim


def train_rnn_basin_aware_superloss(
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	superloss: torch.nn.Module,
	epoch: int,
	device: torch.device,
	**kwargs: Any,
) -> None:
	network.train()

	for batch_idx, (x, y, b) in enumerate(loader):
		assert x.ndim == 3 and y.ndim == 2 and b.ndim == 1
		x = x.to(device)  # (B, T, D)
		y = y.to(device)  # (B, 1)
		b = b.to(device)  # (B,)

		optimizer.zero_grad()
		pred = network(x)
		loss = torch.nn.functional.mse_loss(pred, y, reduction="none")

		sl, w = superloss(loss, b)
		sl.mean().backward()

		optimizer.step()

		print(
			f"\rEpoch {epoch:3d} {numpy.float32(batch_idx+1) / numpy.float32(len(loader)) * 100:3.2f} loss {loss.mean().tolist():.3f}",
			end="",
		)

	print(f"\tsigma {w.mean().tolist():.3f}", end="")


def inference_rnn(
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	network.eval()

	pred = []
	target = []

	with torch.no_grad():
		for x, y, _ in loader:
			assert x.ndim == 3 and y.ndim == 2
			x = x.to(device)
			logit = network(x).cpu().numpy()

			pred.append(logit)
			target.append(y.numpy())

	pred = numpy.concatenate(pred, axis=0).astype(numpy.float32)
	target = numpy.concatenate(target, axis=0).astype(numpy.float32)
	assert pred.shape == target.shape

	return pred, target

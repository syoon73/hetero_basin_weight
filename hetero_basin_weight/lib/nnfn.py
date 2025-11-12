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


class BasinAwareSuperLoss(torch.nn.Module):
	def __init__(
		self,
		target_idx: int,
		n_basins: int,
		sim_thr: float = 0.0,
		sim_scale: float = 1.96,
		lam: float = 1.0,
		mom: float = 0.1,
		mu: float = 0.0,
	) -> None:
		super(BasinAwareSuperLoss, self).__init__()
		assert 0 < lam and 0 <= mu

		#
		self.target_idx = int(target_idx)
		self.n_basins = int(n_basins)

		# Hparams for \sim
		self.sim_thr = float(sim_thr)
		self.sim_scale = float(sim_scale)

		# Hparams for \sigma
		self.lam = float(lam)
		self.mom = float(mom)
		self.mu = float(mu)

		# Automatically estimated
		self.tau = None
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
	) -> Tuple[numpy.ndarray, numpy.ndarray]:
		params = tuple(network.parameters())

		loss_all = []
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

			loss_all.append(loss_b.item())
			grad_all.append(grad_b)

		#
		loss_all = numpy.asarray(loss_all, dtype=numpy.float32)

		#
		grad_all = numpy.vstack(grad_all, dtype=numpy.float32)
		grad_all /= numpy.linalg.norm(grad_all, axis=-1, keepdims=True)
		grad_sim = numpy.matmul(grad_all, grad_all[self.target_idx])
		grad_sim[self.target_idx] = 1.0

		return loss_all, grad_sim

	def _compute_gradient_similarity(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_basin_batches: int = 0,
	) -> Tuple[numpy.ndarray, numpy.ndarray]:
		#
		network.eval()
		### For `cudnn RNN backward` ###
		if hasattr(network, "rnn"):
			assert isinstance(network.rnn, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU))
			network.rnn.train()
		################################

		return self._compute_gradient_similarity_sequential(
			network, loader_per_basin, device, num_basin_batches=num_basin_batches
		)

	def compute_sigma(
		self,
		network: torch.nn.Module,
		loader_per_basin: List[torch.utils.data.DataLoader],
		device: torch.device,
		num_basin_batches: int = 0,
	) -> numpy.ndarray:
		#
		loss_all, grad_sim = self._compute_gradient_similarity(
			network,
			loader_per_basin,
			device,
			num_basin_batches=num_basin_batches,
		)

		#
		if self.sim_thr == 0:
			grad_sim_mod = grad_sim
		else:
			# [-1, c, 1] --> [-1-c, c-c==0, 1-c] --> [-(1+c)/(1-c), 0, 1]
			grad_sim_mod = (grad_sim - self.sim_thr) / (1.0 - self.sim_thr)
		tau_all = loss_all.mean() + self.sim_scale * grad_sim_mod

		if self.tau is not None and self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * tau_all
		else:
			self.tau = tau_all

		#
		eps = numpy.finfo(numpy.float32).eps

		beta = (loss_all - self.tau) / self.lam
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

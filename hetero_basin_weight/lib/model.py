import math
import torch
import torch.nn as nn
from typing import Any


class RNN(nn.Module):
	def __init__(
		self,
		input_size: int,
		output_size: int,
		time_embed: bool = False,
		hidden_size: int = 256,
		num_rnn_layers: int = 1,
		num_regression_layers: int = 1,
		rnn_type: str = "lstm",
		bidirectional: bool = False,
		input_dropout: float = 0,
		rnn_dropout: float = 0,
		regression_dropout: float = 0,
		**kwargs: Any
	) -> None:
		super(RNN, self).__init__()
		self.input_size = int(input_size)

		# Time embedding (optional)
		if time_embed:
			self.time_embed = nn.Linear(3, hidden_size)
		else:
			self.register_module("time_embed", None)

		self.rnn_type = str(rnn_type)
		rnn_class = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[self.rnn_type]
		assert self.input_size > 0 and num_rnn_layers > 0 and num_regression_layers > 0

		# Input projection
		self.input_proj = nn.Linear(self.input_size, hidden_size)
		if 0 < input_dropout < 1:
			self.input_proj_dropout = nn.Dropout(float(input_dropout))
		else:
			self.register_module("input_proj_dropout", None)

		# RNN
		self.rnn = rnn_class(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=num_rnn_layers,
			bias=True,
			batch_first=True,
			dropout=rnn_dropout,
			bidirectional=bidirectional,
		)

		#
		n_emb = 2 if bidirectional else 1

		# Regression
		regression = []
		for i in range(num_regression_layers):
			n_in = hidden_size * n_emb if i == 0 else hidden_size
			n_out = output_size if i == num_regression_layers - 1 else hidden_size

			if 0 < regression_dropout < 1:
				regression.append(nn.Dropout(regression_dropout))

			regression.append(nn.Linear(n_in, n_out))

			if i < num_regression_layers - 1:
				regression.append(nn.GELU())

		self.regression = nn.Sequential(*regression)

		#
		self.init_rnn_params()
		self.rnn.flatten_parameters()

	def init_rnn_params(self) -> None:
		# rnn
		m = self.rnn
		assert isinstance(m, (nn.GRU, nn.LSTM, nn.RNN))
		for name, param in m.named_parameters():
			if "weight_hh" in name or "weight_hr" in name:
				gain = 1.0 / math.sqrt(m.hidden_size)
				nn.init.orthogonal_(param, gain=gain)
			elif "bias" in name:
				if "bias_hh" in name:
					if isinstance(m, nn.LSTM):
						nn.init.zeros_(param[0])
						nn.init.ones_(param[1])
						nn.init.zeros_(param[2:])
					elif isinstance(m, nn.GRU):
						nn.init.ones_(param[0])
						nn.init.zeros_(param[1:])
					else:
						nn.init.zeros_(param)
				else:
					nn.init.zeros_(param)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, D)
		if self.time_embed is not None:
			assert x.size(-1) == 3 + self.input_size
			t, x = torch.split(x, [3, self.input_size], dim=-1)
			t = self.time_embed(t)

		# Projection
		assert x.size(-1) == self.input_size
		v = nn.functional.gelu(self.input_proj(x))
		if self.input_proj_dropout is not None:
			v = self.input_proj_dropout(v)

		# Encoding
		if self.time_embed is not None:
			v = v + t
		h, _ = self.rnn(v)
		h = h[:, -1]

		# Regression
		y = self.regression(h)

		return y


if __name__ == "__main__":
	torch.manual_seed(0)
	B = 3
	T = 10
	D1, D2 = 2, 4
	time_embde = False

	x = torch.rand(B, T, D1 + (3 if time_embde else 0))
	z = torch.rand(B, D2)
	x = torch.cat([x, z.unsqueeze(1).broadcast_to((B, T, D2))], dim=-1)

	m = RNN(D1 + D2, 1, time_embed=time_embde)

	y = m(x)

	print(y)
	print(m.input_proj.bias)
	# print(y.shape)

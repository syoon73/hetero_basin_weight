import numpy
from scipy.stats import pearsonr


def nse_loss(
	pred: numpy.ndarray, target: numpy.ndarray, axis: int = 0, eps: float = 1e-02
) -> numpy.ndarray:
	# Nash-Sutcliffe model efficiency coefficient
	if axis < 0:
		axis += pred.ndim
	assert (
		pred.ndim == target.ndim
		and 0 <= axis < pred.ndim
		and pred.shape[axis] == target.shape[axis]
	)

	if pred.shape[axis] < 2:
		return numpy.full(
			[v for i, v in enumerate(pred.shape) if i != axis],
			numpy.nan,
			dtype=numpy.float32,
		)

	numer = numpy.nansum((target - pred) ** 2, axis=axis)
	denom = numpy.nansum(
		(target - numpy.nanmean(target, axis=axis, keepdims=True)) ** 2, axis=axis
	)
	return 1.0 - (numer / numpy.maximum(denom, eps))


def kge_loss(
	pred: numpy.ndarray, target: numpy.ndarray, axis: int = 0, eps: float = 1e-02
) -> numpy.ndarray:
	# Kling-Gupta Efficiency
	if axis < 0:
		axis += pred.ndim
	assert (
		pred.ndim == target.ndim
		and 0 <= axis < pred.ndim <= 2
		and pred.shape[axis] == target.shape[axis]
	)

	if pred.shape[axis] < 2:
		return numpy.full(
			[v for i, v in enumerate(pred.shape) if i != axis],
			numpy.nan,
			dtype=numpy.float32,
		)

	pred_mean = numpy.nanmean(pred, axis=axis)
	pred_std = numpy.nanstd(pred, axis=axis)

	target_mean = numpy.nanmean(target, axis=axis)
	target_std = numpy.nanstd(target, axis=axis)

	if pred.ndim == 1:
		r, _ = pearsonr(target, pred)
	else:
		remained_axis = 1 if axis == 0 else 0
		if pred.shape[remained_axis] == 1:
			r, _ = pearsonr(target.squeeze(remained_axis), pred.squeeze(remained_axis))

		else:
			vfunc = numpy.vectorize(pearsonr, signature="(n),(n)->(),()")

			if axis == 0:
				r, _ = vfunc(target.T, pred.T)
			else:
				r, _ = vfunc(target, pred)

	alpha = pred_std / numpy.maximum(target_std, eps)
	beta = pred_mean / numpy.maximum(target_mean, eps)

	return 1.0 - numpy.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

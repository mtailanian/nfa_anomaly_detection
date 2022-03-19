import math
import torch
import numpy as np
import scipy.stats as st
from typing import Tuple, List
import torch.nn.functional as F


def compute_region_log_nfa(distances: List[np.array], threshold: float):
	"""
	distances: list of `n_scales` torch.Tensor of size [n_channels, n_filters, height, width]
	"""
	n_scales = len(distances)
	log_nfa_by_scale = []
	for scale in range(n_scales):
		s_lnfa = compute_region_log_nfa_single_scale_original_size(distances[scale], threshold)
		log_nfa_by_scale.append(s_lnfa)
	log_nfa_out = np.min(np.array(log_nfa_by_scale), axis=0)
	return log_nfa_out


def compute_region_log_nfa_single_scale_original_size(distance, threshold):
	distance = F.interpolate(torch.Tensor(distance), size=(256, 256), mode='bicubic', align_corners=True).numpy()
	_, n_filters, height, width = distance.shape

	log_nfa_map = np.zeros((n_filters, height, width), dtype=np.float32)
	used_pixels = np.zeros((n_filters, height, width), dtype=bool)
	for f in range(n_filters):
		d = distance[0, f]
		for i in range(height):
			for j in range(width):
				if d[i, j] >= threshold:

					region, score, d = compute_neighbors(d, (i, j), threshold)
					log_nfa_value = compute_region_log_nfa_value(region, score, height, width)

					log_nfa_map[f][tuple(zip(*region))] = log_nfa_value
					used_pixels[f][tuple(zip(*region))] = True

	log_nfa_map[np.logical_not(used_pixels)] = np.max(log_nfa_map)
	min_log_nfa_map = np.min(log_nfa_map, axis=0)
	return min_log_nfa_map


def compute_region_log_nfa_value(region, score, height, width):
	"""
	The number of blocks to be tested is HW/f^2, being f the independency factor
	Then we must multiply by al possible subsamplings, which adds a f^2 multiplying
	So the final number of tests is HW * #Polyominoes of the sub-sampled equivalent size
	"""
	scale = 0
	filter_size = 3 * 2 ** (2 + scale)
	independency_factor = filter_size ** 2

	region_size = len(region)

	log_number_of_tests = \
		np.log(height) + np.log(width) - np.log(independency_factor) + \
		get_log_number_of_possible_polyominoes(region_size / independency_factor)

	chi_probability = 1 - st.chi2.cdf(score / independency_factor, region_size / independency_factor)
	chi_log_probability = np.log(np.max([chi_probability, np.finfo(float).eps]))

	log_nfa_value = log_number_of_tests + chi_log_probability
	return log_nfa_value


def get_log_number_of_possible_polyominoes(region_size):
	alpha = 0.316915
	beta = 4.062570
	return np.log(alpha) + region_size * np.log(beta) - np.log(region_size)


def compute_neighbors(distance: np.array, seed: Tuple[int, int], threshold: float):
	"""
	Algorithm 3 from https://www.ipol.im/pub/art/2021/342/article_lr.pdf
	@param distance: (error epsilon in reference). Matrix of image size, containing the mahalanobis distance (error in
		gradient orientations in ref.) TODO: normalized?
	@param seed: seed pixel. Tuple containing the indexes (y, x)
	@param threshold: (rho parameter in reference). Threshold to add the neighbor pixel to the region
	@return:
	"""
	assert isinstance(distance, np.ndarray), "distance parameter must be a np.array"
	assert len(distance.shape) == 2, "distance parameter must be two dimensional array"
	assert isinstance(seed, Tuple), "seed parameter must be a tuple"

	height, width = distance.shape

	region = [seed]
	score = distance[seed]
	distance[seed] = 0

	has_region_grew = True
	while has_region_grew:
		has_region_grew = False
		for pixel in region:
			neighborhood = get_4_connectivity_neighborhood(pixel, height, width)
			for neighbor in neighborhood:
				if distance[neighbor] >= threshold:
					region.append(neighbor)
					score += distance[neighbor]
					distance[neighbor] = 0
					has_region_grew = True

	return region, score, distance


def get_4_connectivity_neighborhood(pixel, height, width):

	def _append_if_inside(point):
		if (point[0] >= 0) and (point[0] < height) and (point[1] >= 0) and (point[1] < width):
			return point

	left = (pixel[0], pixel[1] - 1)
	right = (pixel[0], pixel[1] + 1)
	top = (pixel[0] - 1, pixel[1])
	bottom = (pixel[0] + 1, pixel[1])

	return list(filter(None, map(_append_if_inside, [left, right, top, bottom])))


def unit_test():
	distance = np.array([
		[0.8, 0.1, 0.1, 0.3],
		[0.6, 0.7, 0.4, 0.1],
		[0.1, 0.6, 0.3, 0.2]
	])
	region, score, modified_distance = compute_neighbors(distance, seed=(0, 0), threshold=0.5)

	region_mask = np.zeros_like(distance)
	region_mask[tuple(zip(*region))] = 1

	assert (region_mask == np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 0]])).all(), "Wrong mask result"
	assert math.isclose(score, 2.7), "Wrong score"

	print("Test passed. Successfully executed")


if __name__ == '__main__':
	unit_test()

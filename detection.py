import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from sklearn.decomposition import PCA
from scipy.interpolate import NearestNDInterpolator

from region_nfa import compute_region_log_nfa


class NFAResnetDetector:
	@classmethod
	def from_command_line_args(cls, args):
		return cls(args.resnet_input_size, args.n_components, args.log_nfa_threshold, args.distance_threshold)

	def __init__(self, resnet_input_size, n_components, log_nfa_threshold=0, distance_threshold=0.5):
		self.resnet_input_size = resnet_input_size
		self.n_components = n_components
		self.log_nfa_thr = log_nfa_threshold
		self.distance_thr = distance_threshold
		self.activation = {}
		self.resnet = self._get_resnet()

	def _get_resnet(self):
		def get_activation(name):
			def hook(model, input, output):
				self.activation[name] = output.detach()
			return hook

		resnet = models.resnet50(pretrained=True)
		for p in resnet.parameters():
			p.requires_grad = False

		bottleneck = 0
		resnet.layer1[bottleneck].conv3.register_forward_hook(get_activation('layer1'))

		return resnet

	def detect(self, img):
		activations = self._get_resnet_response(img)
		decorrelated_activations = self._decorrelate_channels(activations['layer1'], self.n_components)
		distance = self._compute_squared_mahalanobis_distance(decorrelated_activations)
		log_nfa = compute_region_log_nfa([distance.numpy()], threshold=self.distance_thr)
		log_nfa = self._post_process(img, log_nfa)
		detection = log_nfa < self.log_nfa_thr
		return log_nfa, detection

	def _get_resnet_response(self, img):
		self.activation = {}

		# Reshape input image to match resnet input size
		if self.resnet_input_size:
			img = F.interpolate(img, (self.resnet_input_size, self.resnet_input_size))

		# Border condition: reflect before network forward pass.
		offset = self.resnet_input_size // 8  # in pixels, related to the resnet input size
		offset_at_l1 = int(offset / 2 / 2)  # Offset at layer 1, after down-sampling 2 times inside ResNet
		img_to_filter = F.pad(img - img.mean(), pad=4 * [offset], mode='reflect')
		_ = self.resnet(img_to_filter)
		for key in self.activation.keys():
			self.activation[key] = self.activation[key][..., offset_at_l1:-offset_at_l1, offset_at_l1:-offset_at_l1]

		return self.activation

	@staticmethod
	def _decorrelate_channels(image, n_comps):
		"""
		Method to decorrelate the channels of an input volume
		:param image: input volume of size [1, n_channels, height, width], in torch format
		:param n_comps: number of components to keep or percentage of explained variance
		:return: output volume of the same width and height and possibly different number of channels
		"""
		x = image[0].reshape(image.shape[1], -1).numpy()
		x -= x.mean()
		pca = PCA(n_comps)
		x_pca = pca.fit_transform(x.transpose())
		# Normalize
		x_pca -= np.min(x_pca, axis=0)
		x_pca /= np.max(x_pca, axis=0)
		return torch.from_numpy(x_pca.transpose()).unsqueeze(0).view((1, -1, image.shape[2], image.shape[3]))

	@staticmethod
	def _compute_squared_mahalanobis_distance(activations):
		activations_mean = torch.mean(activations, dim=(2, 3), keepdim=True)
		activations_std = torch.std(activations, dim=(2, 3), keepdim=True)
		distance = (activations - activations_mean) ** 2 / activations_std ** 2  # [n_chann, n_filters, height, width]
		return distance

	@staticmethod
	def _post_process(img, log_nfa):
		# Resize to the original size
		out_log_nfa = F.interpolate(torch.Tensor(log_nfa).unsqueeze(0).unsqueeze(0), img.shape[2:])[0, 0].numpy()
		# If any NaNs, replace them with the closest value
		mask = np.where(~np.isnan(out_log_nfa))
		interp = NearestNDInterpolator(np.transpose(mask), out_log_nfa[mask])
		out_log_nfa = interp(*np.indices(out_log_nfa.shape))
		# If any -Inf, replace them with the lowest value != -Inf
		lowest_value = np.min(out_log_nfa[out_log_nfa > -np.inf])
		out_log_nfa[out_log_nfa == -np.inf] = lowest_value
		return out_log_nfa

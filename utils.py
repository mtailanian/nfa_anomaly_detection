import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def load_image(image_path):
	image_transform = transforms.Compose([transforms.ToTensor()])
	image = Image.open(image_path)
	image = image_transform(image).unsqueeze(0)
	return image


def show_results(input_image: torch.Tensor, log_nfa: np.array, detection: np.array):
	plt.figure(figsize=(15, 5))

	plt.subplot(131)
	plt.imshow(input_image[0].permute(1, 2, 0).numpy())
	plt.axis('off')

	plt.subplot(132)
	plt.imshow(log_nfa)
	plt.axis('off')

	plt.subplot(133)
	plt.imshow(detection)
	plt.axis('off')

	plt.tight_layout()
	plt.show()

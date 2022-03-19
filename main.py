import argparse
from typing import Union

from detection import NFAResnetDetector
from utils import load_image, show_results


def parse_args():
	parser = argparse.ArgumentParser(
		description='Anomaly detector, based on the computation of the Number of False Alarms (NFA) over regions.'
	)
	parser.add_argument('image_path', type=str, help='Path of the image to process')
	parser.add_argument(
		'-thr',
		'--log_nfa_threshold',
		type=float,
		default=0,
		dest='log_nfa_threshold',
		help='Threshold over the computed NFA map, for final segmentation.'
	)
	parser.add_argument(
		'-dist_thr',
		'--distance_threshold',
		type=float,
		default=0.5,
		dest='distance_threshold',
		help='Threshold over the squared Mahalanobis distances, for computing the candidate regions.'
	)
	# TODO: Check if 32 is correct
	parser.add_argument(
		'-s', '--size',
		type=int,
		default=256,
		help='Input size for ResNet. Must be divisible by 32.',
		dest='resnet_input_size'
	)
	parser.add_argument(
		'-pca', '--pca_std',
		type=Union[int, float],
		default=35,
		help='If float: the percentage of the variance to keep in PCA. If int: the number of components to keep.',
		dest='n_components'
	)
	arguments = parser.parse_args()
	assert arguments.resnet_input_size % 32 == 0, "Resnet input size must be divisible by 32"
	return arguments


if __name__ == '__main__':
	args = parse_args()
	print("Arguments: ", args)
	image = load_image(args.image_path)
	print("Creating detector...")
	detector = NFAResnetDetector.from_command_line_args(args)
	print("\tDone!\nPerforming detection...")
	log_nfa, detection = detector.detect(image)
	print("\tDone!\nShowing result...")
	show_results(image, log_nfa, detection)
	print("\tDone!")

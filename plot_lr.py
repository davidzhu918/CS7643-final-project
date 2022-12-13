import matplotlib
import json

import matplotlib.pyplot as plt

PATH_TO_PARENT_DIR = './experiments/panoptic_deeplab/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv'


def main():
	path_to_json_dir = 'variant_resnet_depth_101_freeze_resnet_layer_4_batch_8_iters_10k'
	json_path = f'{PATH_TO_PARENT_DIR}/{path_to_json_dir}/metrics.json'

	curr_itr = -1
	curr_losses = []
	curr_iters = []
	for line in open(json_path, 'r'):
		raw_dict = json.loads(line)
		if 'total_loss' in raw_dict:
			iter = raw_dict['iteration']
			if iter < curr_itr:
				break
			loss = raw_dict['total_loss']
			curr_losses.append(loss)
			curr_iters.append(iter)
			curr_itr = iter

	print(len(curr_losses))

	plt.plot(curr_iters, curr_losses)
	plt.title("Validation Loss")
	plt.savefig('./visualization_images/panoptic_deeplab_lr.png')
	plt.show()



if __name__ == '__main__':
	main()
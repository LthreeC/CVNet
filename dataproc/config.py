# sampling.py
TARGET_COUNT = 2000
INPUT_FOLDER = '../datasets/ts_data2/train'
OUTPUT_FOLDER = '../datasets/ts_data2/train_sampled'

# calc_mean_std.py
IMAGE_FOLDER = '../datasets/ts_data'

# transforms.py
Mean = [0.5, 0.5, 0.5]
Std = [0.5, 0.5, 0.5]

# split_folder
TRAIN_RATIO = .7
VALID_RATIO = .1
TEST_RATIO = .2
ORI_INPUT_PATH = '../datasets/braintumor'
SPT_OUT_PATH = '../datasets/ts_data'

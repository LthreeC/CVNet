import splitfolders

TRAIN_RATIO = .7
VALID_RATIO = .1
TEST_RATIO = .2
ORI_INPUT_PATH = '../datasets/merged'
SPT_OUT_PATH = '../datasets/merged_split'

if __name__ == '__main__':
    splitfolders.ratio(input=ORI_INPUT_PATH, output=SPT_OUT_PATH, seed=3, ratio=(TRAIN_RATIO, VALID_RATIO, TEST_RATIO)) # default values
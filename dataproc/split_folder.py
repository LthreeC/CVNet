import splitfolders
from dataproc.config import ORI_INPUT_PATH, SPT_OUT_PATH, TRAIN_RATIO, VALID_RATIO, TEST_RATIO

if __name__ == '__main__':
    splitfolders.ratio(input=ORI_INPUT_PATH, output=SPT_OUT_PATH, seed=3, ratio=(TRAIN_RATIO, VALID_RATIO, TEST_RATIO)) # default values
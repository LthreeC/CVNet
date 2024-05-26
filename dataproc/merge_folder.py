import os
import shutil
from tqdm import tqdm


def merge_same_named_folders(src_dir1, src_dir2, output_dir):
    """
    :param src_dir1
    :param src_dir2
    :param output_dir
    """
    sub_dirs = [d for d in os.listdir(src_dir1) if os.path.isdir(os.path.join(src_dir1, d))]

    with tqdm(total=len(sub_dirs), desc="Merging folders") as pbar:
        for sub_dir in sub_dirs:
            src_path1 = os.path.join(src_dir1, sub_dir)
            src_path2 = os.path.join(src_dir2, sub_dir)
            output_sub_dir = os.path.join(output_dir, sub_dir)

            os.makedirs(output_sub_dir, exist_ok=True)

            if os.path.exists(src_path1):
                for item in os.listdir(src_path1):
                    src_item_path = os.path.join(src_path1, item)
                    dest_item_path = os.path.join(output_sub_dir, item)

                    if os.path.isdir(src_item_path):
                        if os.path.exists(dest_item_path):
                            merge_directories(src_item_path, dest_item_path)
                        else:
                            shutil.copytree(src_item_path, dest_item_path)
                    else:
                        shutil.copy2(src_item_path, dest_item_path)

            if os.path.exists(src_path2):
                for item in os.listdir(src_path2):
                    src_item_path = os.path.join(src_path2, item)
                    dest_item_path = os.path.join(output_sub_dir, item)

                    if os.path.isdir(src_item_path):
                        if os.path.exists(dest_item_path):
                            merge_directories(src_item_path, dest_item_path)
                        else:
                            shutil.copytree(src_item_path, dest_item_path)
                    else:
                        shutil.copy2(src_item_path, dest_item_path)

            pbar.update(1)


def merge_directories(src_dir, dest_dir):
    """

    :param src_dir:
    :param dest_dir:
    """
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_path):
            if os.path.exists(dest_path):
                merge_directories(src_path, dest_path)
            else:
                shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)


if __name__ == '__main__':
    src_dir1 = '../datasets/train'
    src_dir2 = '../datasets/test'
    output_dir = '../datasets/merged'

    merge_same_named_folders(src_dir1, src_dir2, output_dir)

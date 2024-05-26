import os
import random
import shutil
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

TARGET_COUNT = 1400
INPUT_FOLDER = '../datasets/merged_split/train'
OUTPUT_FOLDER = '../datasets/merged_split/train_sampled'

def oversample_images(input_folder_path, output_folder_path, target_count):
    image_paths = [os.path.join(input_folder_path, img) for img in os.listdir(input_folder_path) if
                   img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_paths)

    transform = transforms.Compose([
        # transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])

    os.makedirs(output_folder_path, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Copying images"):
        shutil.copy(img_path, output_folder_path)

    pbar = tqdm(total=target_count, desc="Augmenting images")
    pbar.update(num_images)
    augmentation_count = 0

    while num_images < target_count:
        img_path = random.choice(image_paths)
        image = Image.open(img_path)
        augmented_image = transform(image)

        augmented_image_path = os.path.join(output_folder_path, f"aug_{augmentation_count}_{os.path.basename(img_path)}")
        augmented_image.save(augmented_image_path)
        num_images += 1
        augmentation_count += 1
        pbar.update(1)

    pbar.close()


def downsample_images(input_folder_path, output_folder_path, target_count):
    image_paths = [os.path.join(input_folder_path, img) for img in os.listdir(input_folder_path) if
                   img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    os.makedirs(output_folder_path, exist_ok=True)

    selected_images = image_paths if len(image_paths) <= target_count else random.sample(image_paths, target_count)

    for img_path in tqdm(selected_images, desc="Downsampling images"):
        shutil.copy(img_path, output_folder_path)


def process_images(input_folder, output_folder, target_count=2000):
    category_folders = os.listdir(input_folder)

    for category_folder in category_folders:
        category_path = os.path.join(input_folder, category_folder)
        output_category_path = os.path.join(output_folder, category_folder)

        if os.path.isdir(category_path):
            num_images = len(
                [img for img in os.listdir(category_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if num_images < target_count:
                oversample_images(category_path, output_category_path, target_count)
            elif num_images > target_count:
                downsample_images(category_path, output_category_path, target_count)
            else:
                # just copy
                shutil.copytree(category_path, output_category_path, dirs_exist_ok=True)


if __name__ == "__main__":
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_COUNT)
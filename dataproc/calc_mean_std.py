import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMAGE_FOLDER = '../datasets/ts_data0/ack'


def calculate_mean_and_std(image_folder):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    means = np.zeros(3)
    stds = np.zeros(3)
    num_images = 0

    transform = transforms.Compose([transforms.ToTensor()])

    total_images = sum(len(files) for _, _, files in os.walk(image_folder) if
                       any(file.lower().endswith(ext) for ext in image_extensions for file in files))

    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
        for root, _, files in os.walk(image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(root, file)
                    try:
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')
                            img_tensor = transform(img)

                            for d in range(3):
                                means[d] += img_tensor[d, :, :].mean().item()
                                stds[d] += img_tensor[d, :, :].std().item()

                            num_images += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                    pbar.update(1)

    means /= num_images
    stds /= num_images

    print(f"Mean: {means}")
    print(f"Standard Deviation: {stds}")
    return means, stds


if __name__ == '__main__':
    calculate_mean_and_std(IMAGE_FOLDER)
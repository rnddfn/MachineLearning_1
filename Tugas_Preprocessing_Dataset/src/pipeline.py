import os
import cv2
import random
import numpy as np
from utils import list_images, save_image
from statsReport import dataset_stats

CLEAN_DATA = "dataset_clean"
FINAL_DATA = "dataset_final"


def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image


def rotate_image(image):
    angle = random.choice([
        random.uniform(-20, -10),
        random.uniform(10, 20)
    ])

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def crop_image(image):
    h, w = image.shape[:2]

    crop_ratio = random.uniform(0.85, 0.95)
    new_h = int(h * crop_ratio)
    new_w = int(w * crop_ratio)

    start_y = random.randint(0, h - new_h)
    start_x = random.randint(0, w - new_w)

    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
    cropped = cv2.resize(cropped, (w, h))

    return cropped


def augment_image(image):
    aug_type = random.choice(["rotate", "crop"])
    if aug_type == "rotate":
        return rotate_image(image), "rot"
    return crop_image(image), "crop"


def copy_clean_to_final():
    for img_path in list_images(CLEAN_DATA):
        image = cv2.imread(img_path)
        if image is None:
            continue

        normalized = normalize_image(image)
        save_ready = (normalized * 255).astype(np.uint8)

        class_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)

        save_path = os.path.join(FINAL_DATA, class_name, file_name)
        save_image(save_path, save_ready)


def augment_all_images(augment_per_image=1):
    for img_path in list_images(CLEAN_DATA):
        image = cv2.imread(img_path)
        if image is None:
            continue

        normalized = normalize_image(image)

        class_name = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for i in range(augment_per_image):
            aug_image, aug_name = augment_image(normalized)
            save_ready = (aug_image * 255).clip(0, 255).astype(np.uint8)

            new_name = f"{base_name}_aug_{aug_name}_{i}.jpg"
            save_path = os.path.join(FINAL_DATA, class_name, new_name)
            save_image(save_path, save_ready)


def main():
    print("Statistik dataset_clean:")
    clean_stats = dataset_stats(CLEAN_DATA)
    for cls, count in clean_stats.items():
        print(cls, ":", count)

    print("\nMenyalin gambar asli ke dataset_final...")
    copy_clean_to_final()

    print("Melakukan augmentasi untuk semua gambar...")
    augment_all_images(augment_per_image=1)

    print("\nStatistik dataset_final:")
    final_stats = dataset_stats(FINAL_DATA)
    for cls, count in final_stats.items():
        print(cls, ":", count)

    print("\nSelesai.")


if __name__ == "__main__":
    main()
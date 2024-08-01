import cv2
import os


def downsample_images_in_directory(input_dir, output_dir, factor):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = cv2.imread(input_path)
            height, width = img.shape[:2]
            new_width = width // factor
            new_height = height // factor
            downsampled_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, downsampled_img)
            print(f"Saved downsampled image to {output_path}")


# 示例用法
input_directory = '/root/waternerf/data/llff/Curasao/enhancewater'
output_directory = '/root/waternerf/data/llff/Curasao/enhancewater_4'
downsample_factor = 4

downsample_images_in_directory(input_directory, output_directory, downsample_factor)

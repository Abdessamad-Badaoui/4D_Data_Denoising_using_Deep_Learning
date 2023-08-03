import os
from skimage import io, img_as_ubyte
from sys import argv

def is_criteria_satisfied(image_path, non_null_threshold, max_value_threshold):
    img = io.imread(image_path)
    non_null_pixels = (img != 0).sum()
    max_value = img.max()
    return non_null_pixels > non_null_threshold and max_value > max_value_threshold

def move_images_by_criteria(source_dir, destination_dir, non_null_threshold, max_value_threshold):
    # if not os.path.exists(destination_dir):
    #     os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(source_dir, filename)
            if is_criteria_satisfied(image_path, non_null_threshold, max_value_threshold):
                destination_path = os.path.join(destination_dir, filename)
                os.rename(image_path, destination_path)


source_directory = argv[1]
destination_directory = argv[2]
threshold_value1 = 50  # Seuil que l'image doit dépasser pour être déplacée
threshold_value2 = 0.1
move_images_by_criteria(source_directory, destination_directory, threshold_value1, threshold_value2)

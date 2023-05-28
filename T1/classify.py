import os
import shutil

def process_images():
    # Create directories for each classification. 
    # Tinyimagenet has 200 classifications, each with 50 images in 'val'.
    def create_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(path)

    def classify_images():
        # Based on 'val_annotations.txt', extract each type of image from 'images'. 
        # Use the information on each line to specify the corresponding directory for storage.
        with open("tiny-imagenet-200/val/val_annotations.txt", 'r') as file:
            for line in file.readlines():
                line = line.strip('\n')
                dir_info = line.split()
                dir_name = dir_info[1:2]
                image_name = dir_info[0:1]
                dir_name_str = dir_name[0][0]
                image_name_str = image_name[0][0]
                image_path = 'tiny-imagenet-200/val/images' + '/' + image_name_str
                dir_path = 'tiny-imagenet-200/val' + '/' + dir_name_str
                shutil.copy(image_path, dir_path)

    # Tinyimagenet provides a txt with 200 classifications, 'wnids.txt', create directories based on it.
    base_dir = 'tiny-imagenet-200/val'
    with open('tiny-imagenet-200/wnids.txt', 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')
            folder_path = base_dir + '/' + line
            create_directory(folder_path)

    classify_images()


# Invoke the function
process_images()
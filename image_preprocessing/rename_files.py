import os
#### Renames files to have a certain prefix


def rename_files(root_dir, new_prefix):
    counter = 1
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            old_path = os.path.join(root, filename)
            new_filename = f"{new_prefix}_{counter}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(root, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")
            counter += 1


root_directory = "C:/Users/Hakan/Desktop/Vision Project/train"

new_prefix = "image"

rename_files(root_directory, new_prefix)

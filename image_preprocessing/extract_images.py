import os
import shutil

#### Extracts Images From individual folders with person names and puts them into their root folder

def move_files_and_remove_dirs(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Iterate over all files in the directory
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                
                if os.path.isfile(file_path):
                    shutil.move(file_path, directory)
            
            if not os.listdir(item_path):
                os.rmdir(item_path)
            else:
                print(f"Directory not empty: {item_path}")

if __name__ == "__main__":
    directory_path = "C:/Users/Hakan/Desktop/Vision Project/archive/lfw-deepfunneled/lfw-deepfunneled/"

    move_files_and_remove_dirs(directory_path)

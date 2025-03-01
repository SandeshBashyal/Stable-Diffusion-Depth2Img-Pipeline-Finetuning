import os
import zipfile

def zip_images_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))

if __name__ == "__main__":
    folder_path = '/home/rmuproject/rmuproject/users/sandesh/Depth-to-Image/Fine-tune-DreamBooth/Output'
    output_zip = '/home/rmuproject/rmuproject/Output.zip'
    zip_images_folder(folder_path, output_zip)
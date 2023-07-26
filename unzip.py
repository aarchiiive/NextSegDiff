import zipfile
from tqdm import tqdm
import glob
import os

def unzip_file(zip_path):
    if 'Part1_Test' in zip_path:
        extract_path = os.path.join(zip_path.split("/")[0], zip_path.split("/")[-1].split(".")[0])
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.infolist())
            progress_bar = tqdm(total=total_files, unit='file')
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_path)
                progress_bar.update(1)

            progress_bar.close()

if __name__ == "__main__":
    # unzip_file('ISIC_2020_Training_Dicom.zip')
    # unzip_file('ISIC_2020_Test_Dicom.zip')
    # unzip_file('ISIC_2020_Training_JPEG.zip')
    # unzip_file('ISIC_2020_Test_JPEG.zip')
    for f in glob.glob("ISIC/zip/*.zip"):
        unzip_file(f)

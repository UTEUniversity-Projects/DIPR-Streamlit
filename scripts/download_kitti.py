import os
import requests
import zipfile
import argparse
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

def download_kitti_3d_dataset(data_dir):
    """Download KITTI 3D dataset"""
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # KITTI 3D Object Detection dataset URLs
    base_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
    label_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
    calib_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
    image_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
    
    files = [
        ("data_object_velodyne.zip", base_url),
        ("data_object_label_2.zip", label_url),
        ("data_object_calib.zip", calib_url),
        ("data_object_image_2.zip", image_url)
    ]
    
    # Download files
    for filename, url in files:
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            download_file(url, filepath)
        else:
            print(f"{filename} already exists")
    
    # Extract files
    print("\nExtracting files...")
    for filename in os.listdir(raw_dir):
        if filename.endswith('.zip'):
            zip_path = os.path.join(raw_dir, filename)
            extract_path = os.path.join(raw_dir, filename[:-4])
            
            if not os.path.exists(extract_path):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_dir)
    
    print("\nDownload complete!")

def main():
    parser = argparse.ArgumentParser(description="Download KITTI 3D dataset")
    parser.add_argument("--data_dir", type=str, default="data/kitti",
                        help="Directory to save KITTI dataset")
    
    args = parser.parse_args()
    download_kitti_3d_dataset(args.data_dir)

if __name__ == "__main__":
    main()
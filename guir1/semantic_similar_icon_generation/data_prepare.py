import os
from datasets import Dataset, Features, Image as ImageFeature, Value
import pandas as pd
from PIL import Image
from io import BytesIO


def inspect_dataset_samples_parquet(
    parquet_path,
    output_dir="inspect_samples",
    n=100000,
    max_dimension=400  # New parameter to control image size
):
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet file using pandas
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet file with pandas: {e}")
        return

    # Convert 'gt_bbox' column to string representation
    df['gt_bbox'] = df['gt_bbox'].apply(lambda x: str(x))

    # Convert pandas DataFrame to datasets.Dataset
    # Define features explicitly to ensure correct types and column names
    features = Features({
        'image': ImageFeature(),
        'instruction': Value('string'),
        'gt_bbox': Value('string'), # Now gt_bbox is a string representation of the list
        'group': Value('string'),
        'ui_type': Value('string')
    })
    # The image column in the pandas DataFrame might be bytes, which needs to be handled
    # Create the dataset from pandas DataFrame, explicitly casting 'image' to ImageFeature
    # Only select the columns that are defined in features
    data = Dataset.from_pandas(df[['image', 'instruction', 'gt_bbox', 'group', 'ui_type']], features=features)


    metadata_log = []
    for i, sample in enumerate(data):
        if i >= n:
            break
        # Save image
        # Access the image data correctly after conversion to datasets.Dataset
        img = sample["image"]


        # Resize the image while maintaining aspect ratio
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)  # High-quality downsampling

        img_save_path = os.path.join(output_dir, f"sample_{i}.jpg")  # Using JPG for smaller files
        img.save(img_save_path, quality=85, optimize=True)  # Save with quality/compression settings

        # Collect and print relevant metadata
        meta = {
            "idx": i,
            "img_file": img_save_path,
            "original_size": img.size,  # Track original dimensions
            "instruction": sample.get("instruction", ""),
            "gt_bbox": sample.get("gt_bbox", ""),
            "group": sample.get("group", ""),
            "ui_type": sample.get("ui_type", ""),
            "other_keys": {k: v for k, v in sample.items() if k not in ("image", "instruction", "gt_bbox", "original_size", "group", "ui_type")}
        }
        metadata_log.append(meta)

    # Save all metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for meta in metadata_log:
            f.write(str(meta) + "\n")
    print(f"Saved {n} samples to {output_dir}/ and metadata.txt.")


# Use the correct filename after download
# parquet_path = "screenspot_test.parquet"
# inspect_dataset_samples_parquet(
#     parquet_path,
#     output_dir="inspect_samples",
#     n=10000,
#     max_dimension=3000  # Adjust this value as needed (smaller = smaller files)
# )



def inspect_dataset_samples_parquet(
    parquet_path,
    output_dir="inspect_samples",
    n=100000,
    max_dimension=400  # New parameter to control image size
):
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet file using pandas
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet file with pandas: {e}")
        return

    # Filter rows where ui_type is 'icon'
    df = df[df['ui_type'] == 'icon']

    if df.empty:
        print("No samples with ui_type='icon' found in the dataset.")
        return

    # Convert 'gt_bbox' column to string representation
    df['gt_bbox'] = df['gt_bbox'].apply(lambda x: str(x))

    # Convert pandas DataFrame to datasets.Dataset
    # Define features explicitly to ensure correct types and column names
    features = Features({
        'image': ImageFeature(),
        'instruction': Value('string'),
        'gt_bbox': Value('string'), # Now gt_bbox is a string representation of the list
        'group': Value('string'),
        'ui_type': Value('string')
    })
    # The image column in the pandas DataFrame might be bytes, which needs to be handled
    # Create the dataset from pandas DataFrame, explicitly casting 'image' to ImageFeature
    # Only select the columns that are defined in features
    # Drop the extra index column before creating the dataset
    data = Dataset.from_pandas(df[['image', 'instruction', 'gt_bbox', 'group', 'ui_type']], features=features, preserve_index=False)

    metadata_log = []
    for i, sample in enumerate(data):
        if i >= n:
            break
        # Save image
        # Access the image data correctly after conversion to datasets.Dataset
        img = sample["image"]

        # Resize the image while maintaining aspect ratio
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)  # High-quality downsampling

        img_save_path = os.path.join(output_dir, f"sample_{i}.jpg")  # Using JPG for smaller files
        img.save(img_save_path, quality=85, optimize=True)  # Save with quality/compression settings

        # Collect and print relevant metadata
        meta = {
            "idx": i,
            "img_file": img_save_path,
            "original_size": img.size,  # Track original dimensions
            "instruction": sample.get("instruction", ""),
            "gt_bbox": sample.get("gt_bbox", ""),
            "group": sample.get("group", ""),
            "ui_type": sample.get("ui_type", ""),
            "other_keys": {k: v for k, v in sample.items() if k not in ("image", "instruction", "gt_bbox", "original_size", "group", "ui_type")}
        }
        metadata_log.append(meta)

    # Save all metadata
    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for meta in metadata_log:
            f.write(str(meta) + "\n")
    print(f"Saved {len(metadata_log)} icon samples to {output_dir}/ and metadata.txt.")


# Use the correct filename after download
parquet_path = "screenspot_test.parquet"
inspect_dataset_samples_parquet(
    parquet_path,
    output_dir="inspect_samples_filtered",
    n=10000,
    max_dimension=100_000  # Adjust this value as needed (smaller = smaller files)
)
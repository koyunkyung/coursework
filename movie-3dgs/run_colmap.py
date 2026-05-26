import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
from PIL import Image


def apply_mask_to_image(image_path: str, mask_path: str, output_path: str):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    # Resize mask if needed
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)
    
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_array] * 3, axis=-1)
    
    white_bg = np.ones_like(img_array) * 255.0
    result = img_array * mask_3ch + white_bg * (1 - mask_3ch)
    
    Image.fromarray(result.astype(np.uint8)).save(output_path, quality=95)


def run_colmap(dataset_path: str, name: str):
    dataset_path = Path(dataset_path)
    input_images = dataset_path / "input"
    database_path = dataset_path / "database.db"
    sparse_path = dataset_path / "sparse"
    output_path = dataset_path / "undistorted"
    
    print(f"\n{'='*60}")
    print(f"Running COLMAP on: {name}")
    print(f"{'='*60}\n")
    
    if not input_images.exists():
        print(f"ERROR: Input folder not found")
        return False
    
    num_images = len(list(input_images.glob("*.jpg")))
    print(f"Found {num_images} images")
    
    # Clean old files
    if database_path.exists():
        database_path.unlink()
    for p in [sparse_path, output_path, dataset_path / "images"]:
        if p.exists():
            shutil.rmtree(p)
    sparse_path.mkdir(exist_ok=True)
    
    # Step 1-3: Feature extraction, matching, mapping
    print("[1/4] Feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(input_images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV"
    ], check=True)
    
    print("[2/4] Feature matching...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", str(database_path),
        "--SequentialMatching.overlap", "10",  # Match with 10 neighboring frames
        "--SequentialMatching.loop_detection", "1"  # Enable loop detection
    ], check=True)
    
    print("[3/4] Sparse reconstruction...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(input_images),
        "--output_path", str(sparse_path)
    ], check=True)
    
    sparse_0 = sparse_path / "0"
    if not sparse_0.exists():
        print("ERROR: No sparse model created")
        return False
    
    # Step 4: Undistortion
    print("[4/4] Image undistortion...")
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(input_images),
        "--input_path", str(sparse_0),
        "--output_path", str(output_path),
        "--output_type", "COLMAP"
    ], check=True)
    
    # Reorganize for 3DGS
    print("[5/5] Organizing files for 3DGS...")
    
    # Move images
    undistorted_images = output_path / "images"
    final_images = dataset_path / "images"
    shutil.move(str(undistorted_images), str(final_images))
    
    # Move sparse model
    undistorted_sparse = output_path / "sparse"
    shutil.rmtree(sparse_path)
    final_sparse = dataset_path / "sparse" / "0"
    final_sparse.mkdir(parents=True, exist_ok=True)
    
    # Copy .bin files
    for f in undistorted_sparse.glob("*.bin"):
        shutil.copy(str(f), str(final_sparse))
    for subdir in undistorted_sparse.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.bin"):
                shutil.copy(str(f), str(final_sparse))
    
    # Cleanup
    shutil.rmtree(output_path)
    
    # Verify
    num_final = len(list(final_images.glob("*.*")))
    bin_files = list(final_sparse.glob("*.bin"))
    print(f"  images/: {num_final} images")
    print(f"  sparse/0/: {[f.name for f in bin_files]}")
    
    required = ["cameras.bin", "images.bin", "points3D.bin"]
    if all((final_sparse / f).exists() for f in required):
        print("  SUCCESS!")
        return True
    return False


def apply_masks_to_colmap_images(name: str):
    colmap_images = Path(f"data/colmap/{name}/images")
    masks_dir = Path(f"data/masks/{name}")
    
    if not masks_dir.exists():
        print(f"No masks found for {name}, skipping mask application")
        return
    
    print(f"\nApplying masks to {name} images...")
    
    # Get mask files
    mask_files = {f.stem: f for f in masks_dir.glob("*.png")}
    
    count = 0
    for img_file in colmap_images.glob("*.jpg"):
        img_stem = img_file.stem
        
        # Find corresponding mask
        if img_stem in mask_files:
            mask_path = mask_files[img_stem]
            apply_mask_to_image(str(img_file), str(mask_path), str(img_file))
            count += 1
    
    print(f"  Applied masks to {count} images")


def setup_input_folders():
    datasets = {
        "scene": "data/frames/scene",
        "teapot": "data/frames/teapot",
        "teddy": "data/frames/teddy",
        "plant": "data/frames/plant"
    }
    
    for name, source in datasets.items():
        input_dir = Path(f"data/colmap/{name}/input")
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(source)
        if source_path.exists():
            count = 0
            for img in source_path.glob("*.jpg"):
                shutil.copy(img, input_dir)
                count += 1
            print(f"{name}: Copied {count} images")


def main():
    print("="*60)
    print("Step 1: Setup input folders (original frames)")
    print("="*60)
    setup_input_folders()
    
    print("\n" + "="*60)
    print("Step 2: Run COLMAP on all datasets")
    print("="*60)
    
    # datasets = ["scene", "teapot", "teddy", "plant"]
    datasets = ["scene"]
    results = {}
    
    for name in datasets:
        try:
            success = run_colmap(f"data/colmap/{name}", name)
            results[name] = success
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = False
    
    print("\n" + "="*60)
    print("Step 3: Apply SAM masks to object images")
    print("="*60)
    
    # Apply masks only to objects (not scene)
    for name in ["teapot", "teddy", "plant"]:
        if results.get(name):
            apply_masks_to_colmap_images(name)
    
    print("\n" + "="*60)
    print("Final Summary")
    print("="*60)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()

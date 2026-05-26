import os
import subprocess
from pathlib import Path


def train_3dgs(data_path: str, output_path: str, name: str, iterations: int = 30000):
    print(f"\n{'='*60}")
    print(f"Training 3DGS: {name}")
    print(f"Data: {data_path}")
    print(f"Output: {output_path}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")
    
    # Verify data structure
    data_path = Path(data_path)
    images_dir = data_path / "images"
    sparse_dir = data_path / "sparse" / "0"
    
    if not images_dir.exists():
        print(f"ERROR: images/ not found in {data_path}")
        return False
    
    if not sparse_dir.exists():
        print(f"ERROR: sparse/0/ not found in {data_path}")
        return False
    
    num_images = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"Found {num_images} images")
    
    # Run training
    cmd = [
        "python", "gaussian-splatting/train.py",
        "-s", str(data_path),
        "-m", output_path,
        "--iterations", str(iterations),
        "--white_background"  # Important for objects with removed background
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {name}")
        return False
    
    # Check output
    ply_file = Path(output_path) / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if ply_file.exists():
        print(f"\nSUCCESS: Model saved to {ply_file}")
        return True
    else:
        print(f"\nWARNING: Expected output not found: {ply_file}")
        return False


def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Datasets to train
    datasets = {
        "scene": {
            "data": "data/colmap/scene",
            "output": "output/scene",
            "iterations": 30000
        },
        "teapot": {
            "data": "data/colmap/teapot",
            "output": "output/teapot",
            "iterations": 30000
        },
        "teddy": {
            "data": "data/colmap/teddy",
            "output": "output/teddy",
            "iterations": 30000
        },
        "plant": {
            "data": "data/colmap/plant",
            "output": "output/plant",
            "iterations": 30000
        }
    }
    
    results = {}
    
    for name, config in datasets.items():
        # Check if data exists
        if not os.path.exists(config["data"]):
            print(f"Skipping {name}: data not found")
            results[name] = False
            continue
        
        success = train_3dgs(
            data_path=config["data"],
            output_path=config["output"],
            name=name,
            iterations=config["iterations"]
        )
        results[name] = success
    
    # Summary
    print("\n" + "="*60)
    print("3DGS Training Summary")
    print("="*60)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    print("\nTrained models location:")
    for name, success in results.items():
        if success:
            print(f"  output/{name}/point_cloud/iteration_30000/point_cloud.ply")


if __name__ == "__main__":
    main()
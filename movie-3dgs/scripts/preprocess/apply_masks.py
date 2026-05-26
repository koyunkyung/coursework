import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def apply_mask_white_bg(image: Image.Image, mask: Image.Image) -> Image.Image:
    
    img_array = np.array(image).astype(np.float32)
    mask_array = np.array(mask).astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_array] * 3, axis=-1)
    
    # White background (255, 255, 255)
    white_bg = np.ones_like(img_array) * 255.0
    
    # Composite: object where mask=1, white where mask=0
    result = img_array * mask_3ch + white_bg * (1 - mask_3ch)
    
    return Image.fromarray(result.astype(np.uint8))


def process_object(obj_name: str):
    
    frames_dir = Path(f'data/frames/{obj_name}')
    masks_dir = Path(f'data/masks/{obj_name}')
    output_dir = Path(f'data/processed/{obj_name}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    print(f"\nProcessing {obj_name}: {len(frame_files)} frames")
    
    for frame_file in tqdm(frame_files, desc=f"Applying masks to {obj_name}"):
        # Load original frame
        frame_path = frames_dir / frame_file
        image = Image.open(frame_path).convert('RGB')
        
        # Load corresponding mask (same name but .png)
        mask_file = frame_file.replace('.jpg', '.png')
        mask_path = masks_dir / mask_file
        
        if not mask_path.exists():
            print(f"Warning: Mask not found for {frame_file}, skipping...")
            continue
        
        mask = Image.open(mask_path).convert('L')
        
        # Apply mask with white background
        result = apply_mask_white_bg(image, mask)
        
        # Save result
        output_path = output_dir / frame_file
        result.save(output_path, quality=95)
    
    print(f"Saved to: {output_dir}")


def main():
    objects = ['teapot', 'teddy', 'plant']
    
    print("="*50)
    print("Applying masks to remove backgrounds")
    print("="*50)
    
    for obj_name in objects:
        if not os.path.exists(f'data/masks/{obj_name}'):
            print(f"Warning: Masks not found for {obj_name}, skipping...")
            continue
        
        process_object(obj_name)
    
    print("\n" + "="*50)
    print("Background removal complete!")
    print("="*50)


if __name__ == '__main__':
    main()
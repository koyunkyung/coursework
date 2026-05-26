import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from sam2.build_sam import build_sam2_video_predictor


def segment_object(predictor, frames_dir: str, output_dir: str, box: list, obj_name: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted frame list
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    num_frames = len(frame_files)
    
    print(f"\n{'='*50}")
    print(f"Processing: {obj_name}")
    print(f"Frames: {num_frames}")
    print(f"Box: {box}")
    print(f"{'='*50}")
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialize state with video frames directory
        state = predictor.init_state(video_path=frames_dir)
        
        # Add bounding box prompt on first frame
        box_array = np.array(box, dtype=np.float32)
        
        _, object_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            box=box_array
        )
        
        print(f"Initial mask generated. Propagating to all frames...")
        
        # Propagate masks through all frames
        for frame_idx, object_ids, masks in tqdm(
            predictor.propagate_in_video(state),
            total=num_frames,
            desc=f"Segmenting {obj_name}"
        ):
            # masks shape: (num_objects, 1, H, W)
            mask = masks[0, 0].cpu().numpy()
            
            # Convert to binary mask (0 or 255)
            binary_mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Save mask with same name as frame
            mask_filename = frame_files[frame_idx].replace('.jpg', '.png')
            mask_path = os.path.join(output_dir, mask_filename)
            Image.fromarray(binary_mask).save(mask_path)
        
        # Reset state for next object
        predictor.reset_state(state)
    
    # Save box info for reproducibility (project requirement)
    info_path = os.path.join(output_dir, 'segment_info.json')
    with open(info_path, 'w') as f:
        json.dump({
            'object': obj_name,
            'box': box,
            'num_frames': num_frames
        }, f, indent=2)
    
    print(f"Masks saved to: {output_dir}")


def main():
    # Your pre-defined bounding boxes
    objects = {
        'teddy': [44, 302, 995, 1660],
        'plant': [10, 733, 1012, 1545],
        'teapot': [290, 702, 1031, 1189]
    }
    
    # Initialize SAM 2 predictor
    print("Loading SAM 2 model...")
    checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    predictor = build_sam2_video_predictor(config, checkpoint)
    print("Model loaded successfully!")
    
    # Process each object
    for obj_name, box in objects.items():
        frames_dir = f"data/frames/{obj_name}"
        masks_dir = f"data/masks/{obj_name}"
        
        if not os.path.exists(frames_dir):
            print(f"Warning: {frames_dir} not found, skipping...")
            continue
        
        segment_object(predictor, frames_dir, masks_dir, box, obj_name)
    
    print("\n" + "="*50)
    print("All segmentation complete!")
    print("="*50)


if __name__ == '__main__':
    main()
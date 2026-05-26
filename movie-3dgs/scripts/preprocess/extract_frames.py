import os
import subprocess
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int = 2, skip_seconds: float = 5.0):
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg',
        '-ss', str(skip_seconds),    # Skip first N seconds
        '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',                 # High quality
        '-start_number', '0',
        os.path.join(output_dir, '%05d.jpg')
    ]
    
    print(f"Extracting: {video_path} -> {output_dir}")
    subprocess.run(cmd, check=True)
    
    # Count extracted frames
    num_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    print(f"  Extracted {num_frames} frames\n")


def main():
    raw_dir = Path('data/raw')
    frames_dir = Path('data/frames')
    
    # Video settings: (fps, skip_seconds)
    videos = {
        'scene': (2, 5.0),     # Scene: 3fps
        # 'teapot': (3, 5.0),    # Objects: 3fps for more detail
        # 'teddy': (3, 5.0),
        # 'plant': (3, 5.0)
    }
    
    for name, (fps, skip) in videos.items():
        video_path = raw_dir / f'{name}.mp4'
        output_dir = frames_dir / name
        
        if video_path.exists():
            extract_frames(str(video_path), str(output_dir), fps=fps, skip_seconds=skip)
        else:
            print(f"Warning: {video_path} not found")


if __name__ == '__main__':
    main()
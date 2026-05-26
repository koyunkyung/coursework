import torch
import numpy as np
import os
import json
import math
from pathlib import Path
from tqdm import tqdm
import torchvision
from argparse import ArgumentParser, Namespace
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

try:
    from scipy.spatial.transform import Rotation
except:
    print("Need scipy: pip install scipy")
    exit(1)

def get_lookat_matrix(camera_pos, target_pos, up=np.array([0, 1, 0])):
    camera_pos = np.array(camera_pos)
    target_pos = np.array(target_pos)
    z_axis = camera_pos - target_pos
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6: return np.eye(3)
    z_axis = z_axis / norm_z
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.vstack([x_axis, y_axis, z_axis])

class Camera:
    def __init__(self, id, width, height, position, rotation_matrix, fx, fy):
        self.id = id
        self.image_width = width
        self.image_height = height
        self.position = np.array(position, dtype=np.float32)
        self.R = np.array(rotation_matrix, dtype=np.float32)
        self.fx = fx
        self.fy = fy
        self.FoVx = 2 * math.atan(width / (2 * fx))
        self.FoVy = 2 * math.atan(height / (2 * fy))
        self.znear = 0.01
        self.zfar = 100.0
        self._setup_matrices()
    
    def _setup_matrices(self):
        device = torch.device('cuda')
        Rt = np.zeros((4, 4), dtype=np.float32)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = -self.R @ self.position
        Rt[3, 3] = 1.0
        # Rt[1, :] *= -1.0 
        Rt[2, :] *= -1.0
        
        tanHalfFovY = math.tan(self.FoVy / 2)
        tanHalfFovX = math.tan(self.FoVx / 2)
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 1.0 / tanHalfFovX
        P[1, 1] = 1.0 / tanHalfFovY
        P[2, 2] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        P[2, 3] = -(2.0 * self.zfar * self.znear) / (self.zfar - self.znear)
        P[3, 2] = -1.0
        
        full_proj = P @ Rt
        self.world_view_transform = torch.tensor(Rt.T, dtype=torch.float32, device=device)
        self.full_proj_transform = torch.tensor(full_proj.T, dtype=torch.float32, device=device)
        self.camera_center = torch.tensor(self.position, dtype=torch.float32, device=device)

def transform_gaussians_synced(gaussians, translate, rotate_deg, scale, center_first=True):
    """
    compose_scene.py의 transform_supersplat 로직과 100% 동일하게 동작하도록 구현
    """
    with torch.no_grad():
        xyz = gaussians._xyz.clone()
        device = xyz.device
        
        # 1. Center at origin (Mean Centroid)
        # compose_scene.py는 mean을 사용하므로 여기서도 mean 사용 (median 아님)
        if center_first:
            centroid = xyz.mean(dim=0)
            xyz = xyz - centroid
        
        # 2. Scale
        xyz = xyz * scale
        
        # 3. Rotate (Euler XYZ)
        if any(r != 0 for r in rotate_deg):
            # Scipy Rotation과 동일한 로직 적용
            R_scipy = Rotation.from_euler('xyz', rotate_deg, degrees=True)
            R_matrix = torch.tensor(R_scipy.as_matrix(), dtype=torch.float32, device=device)
            
            # Position Rotation
            xyz = xyz @ R_matrix.T
            
            # Quaternion Rotation
            # 3DGS stores [w, x, y, z] -> but we manipulate as [x, y, z, w] for scipy logic logic if needed
            # But standard rotation of quat: q_new = R * q_old
            rots = gaussians._rotation.clone()
            # 3DGS: [w, x, y, z] (real first)
            # Scipy: [x, y, z, w] (real last)
            
            # Convert to [x,y,z,w] for scipy rotation logic
            q_xyzw = rots[:, [1, 2, 3, 0]].cpu().numpy()
            
            # Apply rotation
            new_q = (R_scipy * Rotation.from_quat(q_xyzw)).as_quat()
            
            # Convert back to [w,x,y,z]
            new_q_wxyz = torch.tensor(new_q[:, [3, 0, 1, 2]], dtype=torch.float32, device=device)
            gaussians._rotation = torch.nn.Parameter(new_q_wxyz)

        # 4. Translate
        xyz = xyz + torch.tensor(translate, dtype=torch.float32, device=device)
        
        # Apply positions
        gaussians._xyz = torch.nn.Parameter(xyz)
        
        # Apply Scale log transform
        if scale != 1.0:
            gaussians._scaling = torch.nn.Parameter(gaussians._scaling + math.log(scale))
            
    return gaussians

def merge_gaussians(gaussians_list, sh_degree=3):
    if not gaussians_list: return GaussianModel(sh_degree)
    merged = GaussianModel(sh_degree)
    merged._xyz = torch.nn.Parameter(torch.cat([g._xyz for g in gaussians_list], dim=0))
    merged._features_dc = torch.nn.Parameter(torch.cat([g._features_dc for g in gaussians_list], dim=0))
    merged._features_rest = torch.nn.Parameter(torch.cat([g._features_rest for g in gaussians_list], dim=0))
    merged._scaling = torch.nn.Parameter(torch.cat([g._scaling for g in gaussians_list], dim=0))
    merged._rotation = torch.nn.Parameter(torch.cat([g._rotation for g in gaussians_list], dim=0))
    merged._opacity = torch.nn.Parameter(torch.cat([g._opacity for g in gaussians_list], dim=0))
    merged.active_sh_degree = gaussians_list[0].active_sh_degree
    return merged

class AnimationRenderer:
    def __init__(self, config_path, cameras_json_path, sh_degree=3):
        self.device = torch.device("cuda")
        self.sh_degree = sh_degree
        with open(config_path) as f: self.config = json.load(f)
        
        # Scene Path
        scene_cfg = self.config['scene']
        self.scene_path = scene_cfg['path'] if isinstance(scene_cfg, dict) else scene_cfg
        
        # Objects Paths
        self.object_paths = {obj['name']: obj['path'] for obj in self.config.get('objects', [])}
        
        with open(cameras_json_path) as f:
            train_cams = json.load(f)
            self.ref_cam = train_cams[0]

    def compose_frame(self, scene_state, objects_state):
        gaussians_list = []
        
        # 1. Scene Processing (중요: scene_state 적용 및 center_first=False)
        try:
            scene_g = GaussianModel(self.sh_degree)
            scene_g.load_ply(self.scene_path)
            # Scene은 절대 원점으로 이동시키지 않고 그 자리에서 회전만 적용 (composition.json 로직 따름)
            transform_gaussians_synced(
                scene_g, 
                scene_state['translate'], 
                scene_state['rotate'], 
                scene_state['scale'], 
                center_first=False # Scene은 False
            )
            gaussians_list.append(scene_g)
        except Exception as e:
            print(f"Error loading scene: {e}")

        # 2. Objects Processing
        for name, state in objects_state.items():
            if name in self.object_paths:
                try:
                    obj_g = GaussianModel(self.sh_degree)
                    obj_g.load_ply(self.object_paths[name])
                    # Objects는 중심점 기준으로 이동해야 함 (center_first=True)
                    transform_gaussians_synced(
                        obj_g, 
                        state['translate'], 
                        state['rotate'], 
                        state['scale'], 
                        center_first=True # Objects는 True
                    )
                    gaussians_list.append(obj_g)
                except Exception as e:
                    print(f"Error loading object {name}: {e}")
                    
        return merge_gaussians(gaussians_list, self.sh_degree)
    
    def render_animation(self, animation_path, output_dir, skip=1):
        with open(animation_path) as f: anim = json.load(f)
        total_frames = anim['metadata']['total_frames']
        fps = anim['metadata']['fps']
        
        frames_dir = Path(output_dir) / 'frames'
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        frames_to_render = range(0, total_frames, skip)
        print(f"Rendering {len(frames_to_render)} frames...")
        
        for frame in tqdm(frames_to_render):
            scene_state = anim['scene'][frame]
            objects_state = {n: anim['objects'][n][frame] for n in anim['objects']}
            
            gaussians = self.compose_frame(scene_state, objects_state)
            
            if 'camera' in anim and len(anim['camera']) > frame:
                cam_data = anim['camera'][frame]
                pos = cam_data['position']
                look_at = cam_data['look_at']
                R = get_lookat_matrix(pos, look_at)
                target_width = 1920
                target_height = 1080
                
                camera = Camera(frame, target_width, target_height, pos, R, self.ref_cam['fx'], self.ref_cam['fy'])
            else: continue
            
            with torch.no_grad():
                result = render(camera, gaussians, pipeline, bg_color)
                image = result["render"]
            torchvision.utils.save_image(image, str(frames_dir / f'frame_{frame:04d}.png'))
            
            del gaussians
            torch.cuda.empty_cache()
            
        print(f"ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/output.mp4")

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--animation', required=True)
    parser.add_argument('--cameras', required=True)
    parser.add_argument('--output', default='renders')
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--sh_degree', type=int, default=3)
    args = parser.parse_args()
    
    renderer = AnimationRenderer(args.config, args.cameras, args.sh_degree)
    renderer.render_animation(args.animation, args.output, args.skip)

if __name__ == '__main__':
    main()
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import json
import os
import sys
import shutil


def load_gaussians(path):
    """Load Gaussian Splatting PLY file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")
    
    ply = PlyData.read(path)
    v = ply['vertex']
    props = {name: np.array(v[name]) for name in v.data.dtype.names}
    return props, ply


def get_center(props):
    """Get the centroid of Gaussian positions."""
    return np.array([props['x'].mean(), props['y'].mean(), props['z'].mean()])


def get_bounds(props):
    """Get bounding box of Gaussian positions."""
    return {
        'min': np.array([props['x'].min(), props['y'].min(), props['z'].min()]),
        'max': np.array([props['x'].max(), props['y'].max(), props['z'].max()])
    }


def transform_supersplat(props, translate, rotate_deg, scale, center_first=True):
    """
    Apply transformation matching SuperSplat's behavior.
    
    SuperSplat transformation order:
    1. Center object at origin (optional, for objects)
    2. Apply scale
    3. Apply rotation (Euler XYZ)
    4. Apply translation
    
    Args:
        props: Gaussian properties dictionary
        translate: [x, y, z] final position
        rotate_deg: [rx, ry, rz] rotation in degrees (Euler XYZ)
        scale: Uniform scale factor
        center_first: If True, center object at origin before transform
    
    Returns:
        Transformed properties dictionary
    """
    # Get positions
    pos = np.column_stack([props['x'], props['y'], props['z']])
    n = len(pos)
    
    # Step 1: Optionally center at origin
    if center_first:
        centroid = pos.mean(axis=0)
        pos = pos - centroid
    
    # Step 2: Apply scale
    pos = pos * scale
    
    # Step 3: Apply rotation
    if any(r != 0 for r in rotate_deg):
        R = Rotation.from_euler('xyz', rotate_deg, degrees=True)
        R_matrix = R.as_matrix()
        
        # Rotate positions
        pos = pos @ R_matrix.T
        
        # Rotate Gaussian orientations (quaternions)
        if 'rot_0' in props:
            # 3DGS: [w, x, y, z] in rot_0, rot_1, rot_2, rot_3
            # scipy: [x, y, z, w]
            q_xyzw = np.column_stack([
                props['rot_1'],  # x
                props['rot_2'],  # y
                props['rot_3'],  # z
                props['rot_0']   # w
            ])
            
            old_rot = Rotation.from_quat(q_xyzw)
            new_rot = R * old_rot
            new_q_xyzw = new_rot.as_quat()  # [x, y, z, w]
            
            props['rot_0'] = new_q_xyzw[:, 3].astype(np.float32)  # w
            props['rot_1'] = new_q_xyzw[:, 0].astype(np.float32)  # x
            props['rot_2'] = new_q_xyzw[:, 1].astype(np.float32)  # y
            props['rot_3'] = new_q_xyzw[:, 2].astype(np.float32)  # z
    
    # Step 4: Apply translation
    pos = pos + np.array(translate)
    
    # Update positions
    props['x'] = pos[:, 0].astype(np.float32)
    props['y'] = pos[:, 1].astype(np.float32)
    props['z'] = pos[:, 2].astype(np.float32)
    
    # Update scales (log space)
    if scale != 1.0:
        log_scale = np.log(scale)
        for i in range(3):
            key = f'scale_{i}'
            if key in props:
                props[key] = (props[key] + log_scale).astype(np.float32)
    
    return props


def merge_gaussians(props_list):
    """Merge multiple Gaussian property dictionaries into one."""
    if not props_list:
        return {}
    
    merged = {}
    for key in props_list[0].keys():
        merged[key] = np.concatenate([p[key] for p in props_list])
    
    return merged


def save_ply(props, path, template_ply):
    """Save Gaussian properties to PLY file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    dtype = template_ply['vertex'].data.dtype
    arr = np.zeros(len(props['x']), dtype=dtype)
    
    for k in dtype.names:
        if k in props:
            arr[k] = props[k]
    
    PlyData([PlyElement.describe(arr, 'vertex')]).write(path)
    print(f"[SAVED] {path} ({len(props['x']):,} gaussians)")


def compose(config_path):
    """
    Main composition function.
    """
    print("="*60)
    print("3DGS SCENE COMPOSITION (SuperSplat Compatible)")
    print("="*60)
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    print(f"\nConfig: {config_path}")
    
    all_props = []
    template_ply = None
    
    # =========================================================================
    # Process SCENE (with optional transformation)
    # =========================================================================
    scene_cfg = cfg.get('scene', {})
    
    # Handle both old format (string) and new format (dict)
    if isinstance(scene_cfg, str):
        scene_path = scene_cfg
        scene_translate = [0, 0, 0]
        scene_rotate = [0, 0, 0]
        scene_scale = 1.0
    else:
        scene_path = scene_cfg.get('path', '')
        scene_translate = scene_cfg.get('translate', [0, 0, 0])
        scene_rotate = scene_cfg.get('rotate', [0, 0, 0])
        scene_scale = scene_cfg.get('scale', 1.0)
    
    if scene_path and os.path.exists(scene_path):
        print(f"\n[SCENE] Loading: {scene_path}")
        scene_props, template_ply = load_gaussians(scene_path)
        
        center = get_center(scene_props)
        bounds = get_bounds(scene_props)
        
        print(f"  Gaussians: {len(scene_props['x']):,}")
        print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"  Bounds: X[{bounds['min'][0]:.2f}, {bounds['max'][0]:.2f}] "
              f"Y[{bounds['min'][1]:.2f}, {bounds['max'][1]:.2f}] "
              f"Z[{bounds['min'][2]:.2f}, {bounds['max'][2]:.2f}]")
        
        # Check if scene needs transformation
        needs_transform = (
            any(t != 0 for t in scene_translate) or
            any(r != 0 for r in scene_rotate) or
            scene_scale != 1.0
        )
        
        if needs_transform:
            print(f"  Transform:")
            print(f"    Position: {scene_translate}")
            print(f"    Rotation: {scene_rotate}°")
            print(f"    Scale: {scene_scale}")
            
            # Scene은 center_first=False (원점 기준 회전)
            scene_props = transform_supersplat(
                scene_props, 
                scene_translate, 
                scene_rotate, 
                scene_scale,
                center_first=False  # Scene은 원래 위치에서 회전
            )
            
            center_after = get_center(scene_props)
            print(f"  After: center=({center_after[0]:.3f}, {center_after[1]:.3f}, {center_after[2]:.3f})")
        
        all_props.append(scene_props)
    else:
        print(f"\n[SCENE] Not found or not specified: {scene_path}")
    
    # =========================================================================
    # Process OBJECTS
    # =========================================================================
    print(f"\n[OBJECTS] Processing {len(cfg.get('objects', []))} objects...")
    
    for obj in cfg.get('objects', []):
        name = obj.get('name', 'unnamed')
        path = obj['path']
        translate = obj.get('translate', [0, 0, 0])
        rotate = obj.get('rotate', [0, 0, 0])
        scale = obj.get('scale', 1.0)
        
        print(f"\n  [{name}]")
        print(f"    Path: {path}")
        
        if not os.path.exists(path):
            print(f"    [ERROR] File not found, skipping")
            continue
        
        props, ply = load_gaussians(path)
        if template_ply is None:
            template_ply = ply
        
        center_before = get_center(props)
        print(f"    Before: center=({center_before[0]:.3f}, {center_before[1]:.3f}, {center_before[2]:.3f})")
        
        print(f"    Transform:")
        print(f"      Position: {translate}")
        print(f"      Rotation: {rotate}°")
        print(f"      Scale: {scale}")
        
        # Objects는 center_first=True (물체 중심 기준 변환)
        props = transform_supersplat(
            props, 
            translate, 
            rotate, 
            scale,
            center_first=True  # Object는 중심에서 변환
        )
        
        center_after = get_center(props)
        print(f"    After: center=({center_after[0]:.3f}, {center_after[1]:.3f}, {center_after[2]:.3f})")
        
        all_props.append(props)
    
    # =========================================================================
    # Merge and Save
    # =========================================================================
    if not all_props:
        print("\n[ERROR] No Gaussians to merge!")
        return
    
    print(f"\n[MERGE] Combining {len(all_props)} Gaussian sets...")
    merged = merge_gaussians(all_props)
    print(f"  Total Gaussians: {len(merged['x']):,}")
    
    output_path = cfg.get('output', 'composed_output/composed.ply')
    print(f"\n[OUTPUT] Saving to: {output_path}")
    save_ply(merged, output_path, template_ply)
    
    # Create model directory structure
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = '.'
    
    # Check if output path already has model structure
    if 'point_cloud/iteration_' in output_path:
        # Already in correct structure, just create cfg_args
        model_dir = output_path.split('/point_cloud/')[0]
    else:
        model_dir = os.path.join(output_dir, 'model')
        pc_dir = os.path.join(model_dir, 'point_cloud', 'iteration_30000')
        os.makedirs(pc_dir, exist_ok=True)
        shutil.copy(output_path, os.path.join(pc_dir, 'point_cloud.ply'))
    
    # Create cfg_args
    source_path = cfg.get('source_colmap', 'data/colmap/scene')
    cfg_content = f"Namespace(data_device='cuda', depths='', eval=False, images='images', model_path='{model_dir}', resolution=-1, sh_degree=3, source_path='{source_path}', white_background=False, train_test_exp=False)"
    
    cfg_args_path = os.path.join(model_dir, 'cfg_args')
    with open(cfg_args_path, 'w') as f:
        f.write(cfg_content)
    
    print(f"\n[MODEL] Created: {cfg_args_path}")
    
    # Summary
    print("\n" + "="*60)
    print("COMPOSITION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    compose(sys.argv[1])
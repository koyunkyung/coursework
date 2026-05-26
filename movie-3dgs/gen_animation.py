import json
import numpy as np
import math

FPS = 30
DURATION = 10
TOTAL_FRAMES = FPS * DURATION

# [User's Safe Camera]
START_CAM_POS = [5.436627991928239, 1.978174053997678, -7.392941123635115]

# Scene 설정
SCENE_CONFIG = {
    "translate": [0, 0, 0],
    "rotate": [0, 0, 180],
    "scale": 1.0
}

# 곰돌이 위치 (주인공)
TEDDY_TARGET = [0.45, -0.08, 2.43]
TEDDY_ROTATION = [0, 0, 180]
TEDDY_SCALE = 0.5

# [수정 1] 주전자 위치를 곰돌이 근처로 강제 이동!
# 기존 z=-2.14는 곰돌이(z=2.43)와 4미터 이상 차이 나서 안 보였던 것임.
TEAPOT_START = [-1.0, -0.5, 2.0] 
TEAPOT_ROTATION = [-30, 0, -180]
TEAPOT_SCALE = 0.7

# 파티 위치 (곰돌이 바로 옆)
TEAPOT_PARTY_POS = [TEDDY_TARGET[0] - 0.6, TEDDY_TARGET[1], TEDDY_TARGET[2] + 0.2] 

def lerp(a, b, t):
    if isinstance(a, list): return [lerp(a[i], b[i], t) for i in range(len(a))]
    return float(a + (b - a) * t)

def ease_in_out(t): return t * t * (3 - 2 * t)

def generate_animation():
    animation = {
        "metadata": {"fps": FPS, "total_frames": TOTAL_FRAMES},
        "scene": [],
        "objects": {"teddy": [], "teapot": []},
        "camera": []
    }

    scene_state = SCENE_CONFIG

    # 카메라 궤적 계산용
    start_cam_vec = np.array(START_CAM_POS)
    radius = np.linalg.norm(start_cam_vec)
    # 시작 각도 계산
    start_angle = math.atan2(start_cam_vec[0], start_cam_vec[2])

    for frame in range(TOTAL_FRAMES):
        t_global = frame / TOTAL_FRAMES
        
        # 1. SCENE
        animation["scene"].append(scene_state)

        # 2. OBJECTS
        breath = math.sin(t_global * 10) * 0.02
        animation["objects"]["teddy"].append({
            "translate": TEDDY_TARGET,
            "rotate": TEDDY_ROTATION,
            "scale": TEDDY_SCALE + breath
        })

        move_t = ease_in_out(min(t_global * 1.5, 1.0))
        curr_tea_pos = lerp(TEAPOT_START, TEAPOT_PARTY_POS, move_t)
        spin = move_t * 360 if move_t < 1.0 else 0
        curr_tea_rot = [TEAPOT_ROTATION[0], TEAPOT_ROTATION[1] + spin, TEAPOT_ROTATION[2]]

        animation["objects"]["teapot"].append({
            "translate": curr_tea_pos,
            "rotate": curr_tea_rot,
            "scale": TEAPOT_SCALE
        })

        # 3. CAMERA (크롭 & 회전 문제 해결 핵심)
        orbit_speed = 0.3 # 천천히 회전
        angle = start_angle + (t_global * orbit_speed)
        
        # [수정 2] 줌 인 (Zoom In) = 크롭 효과
        # 0.8 -> 0.35로 줄여서 엄청 가까이 붙입니다. 검은 테두리 제거용.
        zoom_factor = 0.35 
        
        cam_x = TEDDY_TARGET[0] + radius * zoom_factor * math.sin(angle)
        cam_z = TEDDY_TARGET[2] + radius * zoom_factor * math.cos(angle)
        cam_y = START_CAM_POS[1] * 0.5 # 높이도 살짝 낮춤

        animation["camera"].append({
            "position": [cam_x, cam_y, cam_z],
            "look_at": TEDDY_TARGET,
            "fov": 60.0
        })

    with open("animation_final.json", "w") as f:
        json.dump(animation, f, indent=2)
    print("✅ Animation generated: Teapot moved closer & Camera Zoomed In.")

if __name__ == "__main__":
    generate_animation()
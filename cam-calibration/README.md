# Camera Calibration with Zhang's Method

Modern computer vision applications require understanding the geometric relationship between 3D world coordinates and 2D image pixels. Camera calibration determines the projection parameters through intrinsic matrix **K** and extrinsic parameters **[R|T]**. While Tsai's method requires precise 3D calibration objects, Zhang's method simplifies this by using planar patterns like checkerboards captured from multiple viewpoints.

This work implements Zhang's method from scratch and compares it with OpenCV's built-in calibration. The implementation includes homography estimation, parameter initialization, and non-linear refinement. Both methods are evaluated through reprojection errors and undistortion quality.

---

## Execution Flow

### Step 1: Corner Detection
```bash
python data/loader.py
```
Detects chessboard corners from calibration images.

**Output:** `results/corner_detection/` - Visualized corner detection results

### Step 2: Zhang's Calibration
```bash
python zhang_cam.py
```
Three-stage calibration process:
1. Initial linear estimation (DLT)
2. Distortion parameter estimation for enhanced refinement
3. Non-linear refinement (Levenberg-Marquardt)

**Outputs:**
- `results/zhang_calibration.json` - Calibration parameters and reprojection error statistics
- `results/reprojection/zhang/stage1_linear/` - Stage 1 reprojection visualization
- `results/reprojection/zhang/stage2_distortion/` - Stage 2 reprojection visualization
- `results/reprojection/zhang/stage3_refined/` - Stage 3 reprojection visualization
- `results/undistorted/zhang/` - Undistorted images (final parameters from stage 3)

### Step 3: OpenCV Calibration
```bash
python opencv_cam.py
```
Calibration using OpenCV's built-in function for comparison.

**Outputs:**
- `results/opencv_calibration.json` - Calibration parameters and reprojection error statistics
- `results/reprojection/opencv/` - Reprojection visualization
- `results/undistorted/opencv/` - Undistorted images

---

## Project Structure
```
├── data/
│   ├── raw_img/                    
│   └── loader.py                   # Corner detection and data preprocessing
├── zhang_cam.py                    # Zhang's method implementation
├── opencv_cam.py                   # OpenCV calibration implementation
└── results/
    ├── corner_detection/           # Corner detection visualization
    ├── zhang_calibration.json      # Zhang's calibration quantitative results
    ├── opencv_calibration.json     # OpenCV calibration quantitative results
    ├── reprojection/
    │   ├── zhang/                  # Zhang's reprojection visualization
    │   │   ├── stage1_linear/      
    │   │   ├── stage2_distortion/  
    │   │   └── stage3_refined/     
    │   └── opencv/                 # OpenCV reprojection visualization
    └── undistorted/
        ├── zhang/                  # Zhang's undistorted images
        └── opencv/                 # OpenCV undistorted images
```
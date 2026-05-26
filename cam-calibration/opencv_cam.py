import cv2
import numpy as np
import os
from data.loader import DataLoader

class OpenCVCalibration:
    def __init__(self, data):
        self.object_points = data['object_points']
        self.image_points = data['image_points']
        self.image_size = data['image_size']
        self.images = data['images']

    def calibrate(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None
        )
        print("Intrinsic Matrix (K):")
        print(mtx)
        print(f"\nfx = {mtx[0,0]:.4f}")
        print(f"fy = {mtx[1,1]:.4f}")
        print(f"cx = {mtx[0,2]:.4f}")
        print(f"cy = {mtx[1,2]:.4f}")
        
        print("\nDistortion Coefficients:")
        print(f"k1={dist[0,0]:.6f}, k2={dist[0,1]:.6f}, p1={dist[0,2]:.6f}, p2={dist[0,3]:.6f}, k3={dist[0,4]:.6f}")
        
        return {
            'camera_matrix': mtx,
            'dist_coeffs': dist,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
    
    def compute_reprojection_error(self, result):
        print("\n=== Reprojection Error per Image ===")
        errors = []
        for i in range(len(self.object_points)):
            # project points
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i],
                result['rvecs'][i],
                result['tvecs'][i],
                result['camera_matrix'],
                result['dist_coeffs']
            )
            # compute reprojection error
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
            print(f"Image {i+1:2d}: {error:.4f} pixels")
        
        print(f"\nMean: {np.mean(errors):.6f} px | Std: {np.std(errors):.6f} px")
        print(f"Min:  {np.min(errors):.6f} px | Max: {np.max(errors):.6f} px")
        
        return errors
    
    def visualize_reprojection(self, result, output_dir="results/reprojection/opencv"):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(self.images)):
            img = self.images[i].copy()
            
            # compute reprojected points
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i],
                result['rvecs'][i],
                result['tvecs'][i],
                result['camera_matrix'],
                result['dist_coeffs']
            )
            
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        
            # detected corners (red)
            for point in self.image_points[i]:
                cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 0, 255), -1)
            # reprojected points (green)
            for point in imgpoints2:
                cv2.circle(img, tuple(point.ravel().astype(int)), 3, (0, 255, 0), -1)

            cv2.rectangle(img, (5, 5), (550, 140), (0, 0, 0), -1)
            
            # legend
            cv2.putText(img, "Red: Detected", (15, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(img, "Green: Reprojected", (15, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(img, f"Error: {error:.4f} px", (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            save_path = os.path.join(output_dir, f"reproj_{i+1:03d}.jpg")
            cv2.imwrite(save_path, img)
        
        print(f"Saved {len(self.images)} reprojection images")
    
    def undistort_images(self, result, output_dir="results/undistorted/opencv"):
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Undistorted Images to {output_dir}/ ===")
        
        # compute optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            result['camera_matrix'],
            result['dist_coeffs'],
            self.image_size,
            1,
            self.image_size
        )
        
        for i, img in enumerate(self.images):
            dst = cv2.undistort(
                img,
                result['camera_matrix'],
                result['dist_coeffs'],
                None,
                newcameramtx
            )
            comparison = np.hstack([img, dst])
            # add labels
            h, w = img.shape[:2]
            cv2.rectangle(comparison, (5, 5), (450, 70), (0, 0, 0), -1)
            cv2.rectangle(comparison, (w + 5, 5), (w + 350, 70), (0, 0, 0), -1)
            
            cv2.putText(comparison, "Original (Distorted)", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(comparison, "Undistorted", (w + 15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            
            save_path = os.path.join(output_dir, f"undist_{i+1:03d}.jpg")
            cv2.imwrite(save_path, comparison)
        
        print(f"Saved {len(self.images)} undistorted images")
        
    def save_results(self, result, errors, filename="results/opencv_calibration.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        import json
        
        # convert rotation vectors to matrices for readability
        rotation_matrices = []
        for rvec in result['rvecs']:
            rmat, _ = cv2.Rodrigues(rvec)
            rotation_matrices.append(rmat.tolist())
        
        results_dict = {
            'method': 'OpenCV cv2.calibrateCamera',
            'num_images': len(self.images),
            'image_size': {
                'width': self.image_size[0],
                'height': self.image_size[1]
            },
            'intrinsics': {
                'camera_matrix': result['camera_matrix'].tolist(),
                'fx': float(result['camera_matrix'][0, 0]),
                'fy': float(result['camera_matrix'][1, 1]),
                'cx': float(result['camera_matrix'][0, 2]),
                'cy': float(result['camera_matrix'][1, 2])
            },
            'distortion': {
                'coefficients': result['dist_coeffs'].ravel().tolist(),
                'k1': float(result['dist_coeffs'][0, 0]),
                'k2': float(result['dist_coeffs'][0, 1]),
                'p1': float(result['dist_coeffs'][0, 2]),
                'p2': float(result['dist_coeffs'][0, 3]),
                'k3': float(result['dist_coeffs'][0, 4])
            },
            'extrinsics': [
                {
                    'image_id': i + 1,
                    'rotation_vector': result['rvecs'][i].ravel().tolist(),
                    'rotation_matrix': rotation_matrices[i],
                    'translation_vector': result['tvecs'][i].ravel().tolist()
                }
                for i in range(len(result['rvecs']))
            ],
            'reprojection_errors': {
                'per_image': [float(e) for e in errors],
                'statistics': {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors))
                }
            }
        }
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    loader = DataLoader(pattern_size=(13, 9), square_size=20.0)
    data = loader.load_data("data/raw_img")

    # OpenCV Calibration
    calibrator = OpenCVCalibration(data)
    result = calibrator.calibrate()

    # compute and display reprojection errors
    errors = calibrator.compute_reprojection_error(result)
    calibrator.visualize_reprojection(result)
    calibrator.undistort_images(result)
    calibrator.save_results(result, errors)
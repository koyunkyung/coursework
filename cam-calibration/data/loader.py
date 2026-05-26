import cv2
import numpy as np
import glob
import os

## extract data points from checkerboard image ##
class DataLoader:
    def __init__(self, pattern_size=(13, 9), square_size=20.0):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.create_object_points()

    # planar grid points
    def create_object_points(self):
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), 
                       np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 
                              0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp

    def load_data(self, image_dir, output_dir="results/corner_detection"):
        os.makedirs(output_dir, exist_ok=True)
        image_files = self.get_image_files(image_dir)

        ## corner detection ##
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

        object_points = []   # 3D points
        image_points = []    # 2D points
        images = []
        image_size = None

        for i, fname in enumerate(image_files):
            img = cv2.imread(fname)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if image_size is None:
                image_size = (gray.shape[1], gray.shape[0])
            # detect corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
            
            if ret:
                # sub-pixel refinement
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                object_points.append(self.objp)
                image_points.append(corners)
                images.append(img)
                
                # visualization
                img_with_corners = cv2.drawChessboardCorners(
                    img.copy(), self.pattern_size, corners, ret
                )
                base_name = os.path.splitext(os.path.basename(fname))[0]
                output_path = os.path.join(output_dir, f"{base_name}_corners.jpg")
                cv2.imwrite(output_path, img_with_corners)

        print(f"Successfully detected: {len(image_points)}/{len(image_files)}\n")
        
        if len(image_points) == 0:
            raise ValueError("No corners detected in any image!")
        
        return {
            'object_points': object_points,
            'image_points': image_points,
            'image_size': image_size,
            'images': images
        }
    

    def get_image_files(self, image_dir):
        patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(image_dir, pattern)))
        return sorted(files)
    

        
if __name__ == "__main__":
    loader = DataLoader(pattern_size=(13, 9), square_size=20.0)
    data = loader.load_data("data/raw_img")
    print(f"Image Size: {data['image_size']}")
    print(f"Number of valid images: {len(data['images'])}")
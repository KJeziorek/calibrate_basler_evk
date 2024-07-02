import os
import glob
import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate basler camera with events camera')
    parser.add_argument('--pattern_size', default='7x5', type=str)
    parser.add_argument('--num_samples', default=50, type=int)
    return parser.parse_args()

def main(args):
    pattern_size = tuple(map(int, args.pattern_size.split('x')))
    obj_points = []
    img_points1 = []
    img_points2 = []

    objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    images1 = sorted(glob.glob('images/*.png'))
    images2 = sorted(glob.glob('reconstruction/*.png'))

    images1.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    images1.pop(0) # remove from images1 first image with name image_0.png because we don't have previous events
    images2.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    for img1_path, img2_path in zip(images1[:args.num_samples], images2[:args.num_samples]):
        img1 = cv2.imread(img1_path)
        img1 = cv2.resize(img1, (1280, 720))
        img2 = cv2.imread(img2_path)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        ret1, corners1 = cv2.findChessboardCornersSB(gray1, pattern_size, None)
        ret2, corners2 = cv2.findChessboardCornersSB(gray2, pattern_size, None)

        if ret1 and ret2:
            obj_points.append(objp)
            img_points1.append(corners1)
            img_points2.append(corners2)

            img1 = cv2.drawChessboardCorners(img1, pattern_size, corners1, ret1)
            img2 = cv2.drawChessboardCorners(img2, pattern_size, corners2, ret2)
            cv2.imshow('Chessboard Corners - Camera 1', img1)
            cv2.imshow('Chessboard Corners - Camera 2', img2)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    ret1, K1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(obj_points, img_points1, gray1.shape[::-1], None, None)
    ret2, K2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(obj_points, img_points2, gray2.shape[::-1], None, None)

    # remap image 1 to image 2 using homography
    H = cv2.findHomography(img_points1[0], img_points2[0])[0]

    # Save the calibration parameters
    os.makedirs('calibration', exist_ok=True)
    np.savez('calibration/calibration_params.npz', K1=K1, dist_coeffs1=dist_coeffs1, K2=K2, dist_coeffs2=dist_coeffs2, H=H)

    images2 = sorted(glob.glob('reconstruction/events/*.png'))
    images2.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

    for i1, i2 in zip(images1[:args.num_samples], images2[:args.num_samples]):
        image_in = cv2.imread(str(i1))
        image_in = cv2.resize(image_in, (1280, 720))

        # Apply K1 and dist_coeffs1 to image_in
        image_in = cv2.undistort(image_in, K1, dist_coeffs1)

        image_base = cv2.imread(str(i2))
        image_base = cv2.resize(image_base, (1280, 720))

        # Apply K2 and dist_coeffs2 to image_base
        image_base = cv2.undistort(image_base, K2, dist_coeffs2)

        image_out = cv2.warpPerspective(image_in, H, (image_base.shape[1], image_base.shape[0]))

        img_concat = cv2.hconcat([image_out, image_base])
        cv2.imshow('Mapped Image and Base Image', img_concat)

        # add on image_out the image_base with a transparency of 0.5
        cv2.addWeighted(image_out, 1, image_base, 0.5, 0, image_out)
        cv2.imshow('Mapped Image and Base Image with Transparency', image_out)
        cv2.waitKey(100)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    main(args)
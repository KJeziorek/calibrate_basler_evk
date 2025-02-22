import os
import glob
import numpy as np
import cv2
import argparse

from utils.read_events import read_events

def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate Basler camera with events camera')
    parser.add_argument('-e', '--event_file', required=True, type=str, help='Path to the input event file')
    parser.add_argument('-i ', '--image_folder', required=True, type=str, help='Path to the images folder')
    parser.add_argument('-T ', '--duration', default=20, type=int, help='Duration of each event window, in milliseconds')
    return parser.parse_args()

def main(args):
    image_files = sorted(glob.glob(f'{args.image_folder}/*.tif'))
    
    if not image_files:
        print(f"No images found in '{args.image_folder}' directory.")
        return
    
    try:
        image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    except ValueError:
        print("Warning: Unable to sort images properly, ensure filenames end with an index.")
    
    events, triggers, _, _ = read_events(args.event_file)
    
    try:
        npzfile = np.load('calibration/calibration_params.npz')
        K1, dist_coeffs1 = npzfile['K1'], npzfile['dist_coeffs1']
        K2, dist_coeffs2 = npzfile['K2'], npzfile['dist_coeffs2']
        H = npzfile['H']
    except Exception as e:
        print(f"Error loading calibration parameters: {e}")
        return

    for img_path, timestamp in zip(image_files, triggers['t']):
        t_start = np.searchsorted(events['t'], timestamp - args.duration * 1000)
        t_end = np.searchsorted(events['t'], timestamp)
        
        if t_start >= t_end:
            print("Skipping frame due to insufficient event data.")
            continue
        
        x, y, p = events['x'][t_start:t_end], events['y'][t_start:t_end], events['p'][t_start:t_end]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
            continue
        
        img = cv2.resize(img, (1280, 720))
        ev_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        ev_image[y[p == 1], x[p == 1]] = (0, 0, 255)
        ev_image[y[p == 0], x[p == 0]] = (255, 0, 0)
        
        img = cv2.undistort(img, K1, dist_coeffs1)
        ev_image = cv2.undistort(ev_image, K2, dist_coeffs2)
        img_warp = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        
        img_concat = cv2.hconcat([img_warp, ev_image])
        cv2.imshow('Mapped Image and Event Image', img_concat)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    main(args)

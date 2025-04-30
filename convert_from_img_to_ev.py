import numpy as np
import cv2
import argparse
import os
import glob
from utils.read_events import read_events

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualising calibrated events')
    parser.add_argument('-i', '--input_file', required=True, type=str)

    args = parser.parse_args()
    return args


def main(args):
    # Read events from hdf5 file

    calib = np.load('calibration/calibration_params.npz')

    K1 = calib['K1']
    K2 = calib['K2']
    dist_coeffs1 = calib['dist_coeffs1']
    dist_coeffs2 = calib['dist_coeffs2']
    H = calib['H']

    events, triggers, id, ts = read_events(args.input_file)
    for idx in range(len(triggers['t'])-1):
        t_start = np.searchsorted(ts, triggers['t'][idx])
        t_end = np.searchsorted(ts, triggers['t'][idx+1])
        idx_start = id[t_start]
        idx_end = id[t_end]

        x = events['x'][idx_start:idx_end]
        y = events['y'][idx_start:idx_end]
        p = events['p'][idx_start:idx_end]

        # Visualise the events
        ev_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        positive_mask = (p == 1)
        negative_mask = (p != 1)

        ev_img[y[positive_mask], x[positive_mask]] = [0, 0, 255]
        ev_img[y[negative_mask], x[negative_mask]] = [255, 0, 0]

        # Get bbox

        file = os.path.join('data/17.02.2025_Faces/Gawel2', 'image_{}.npy'.format(idx+25))
        bboxes = np.load(file, allow_pickle=True)
        
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]['bbox']

            x1, x2 = x1 * 1280 / 1440, x2 * 1280 / 1440
            y1, y2 = y1 * 720 / 1080, y2 * 720 / 1080
            # —————————————————————————————————————————————
            # 1) build the 4 corners of the original bbox:
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32).reshape(-1, 1, 2)

            # 2) undistort them from the RGB camera (pixel → pixel):
            undist_pix = cv2.undistortPoints(
                corners, K1, dist_coeffs1,
                P=K1
            )  # still in pixel coords, but with distortion removed

            # 3) warp into the event‐camera coordinate frame:
            warped_pix = cv2.perspectiveTransform(
                undist_pix, H
            ).reshape(-1, 2)  # shape = (4,2)

            # 4) turn these warped pixels into *normalized* coords for the event cam:
            fx2, fy2 = K2[0,0], K2[1,1]
            cx2, cy2 = K2[0,2], K2[1,2]
            x_norm = (warped_pix[:,0] - cx2) / fx2
            y_norm = (warped_pix[:,1] - cy2) / fy2

            obj_pts = np.vstack([
                x_norm,
                y_norm,
                np.ones_like(x_norm)
            ]).T.reshape(-1, 1, 3)  # shape = (4,1,3)

            # 5) project through event‐cam intrinsics + distortion:
            img_pts, _ = cv2.projectPoints(
                obj_pts,
                rvec=np.zeros((3,1)),  # no extra rotation
                tvec=np.zeros((3,1)),  # no extra translation
                cameraMatrix=K2,
                distCoeffs=dist_coeffs2
            )
            img_pts = img_pts.reshape(-1, 2)

            # 6) axis‐aligned box in event‐img:
            xs, ys = img_pts[:,0], img_pts[:,1]
            x_ev_min, x_ev_max = int(xs.min()), int(xs.max())
            y_ev_min, y_ev_max = int(ys.min()), int(ys.max())

            # draw it:
            cv2.rectangle(
                ev_img,
                (x_ev_min, y_ev_min),
                (x_ev_max, y_ev_max),
                (0, 255, 0), 2
            )
        # —————————————————————————————————————————————


        # Draw bbox

        # frame = cv2.imread('images/image_{}.png'.format(idx+1))
        # frame = cv2.resize(frame, (1280, 720))
        # frame = cv2.undistort(frame, K1, dist_coeffs1)

        # frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

        # ev_img = cv2.undistort(ev_img, K2, dist_coeffs2)

        # img_concat = cv2.hconcat([frame, ev_img])

        cv2.imshow('Concatenate', ev_img)
        cv2.waitKey(100)


if __name__ == '__main__':
    args = parse_args()
    main(args)
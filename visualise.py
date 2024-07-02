import numpy as np
import cv2
import argparse
import os

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
        if not os.path.exists('images/image_{}.png'.format(idx+1)):
            break
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

        frame = cv2.imread('images/image_{}.png'.format(idx+1))
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.undistort(frame, K1, dist_coeffs1)

        frame = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

        ev_img = cv2.undistort(ev_img, K2, dist_coeffs2)

        img_concat = cv2.hconcat([frame, ev_img])

        cv2.imshow('Concatenate', img_concat)

        cv2.addWeighted(frame, 1, ev_img, 0.5, 0, frame)
        cv2.imshow('Mapped Image and Base Image with Transparency', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    args = parse_args()
    main(args)
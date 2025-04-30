import numpy as np
import cv2

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

def convert_bbox(bboxes, img_shape, ev_img_shape, K1, K2, dist_coeffs1, dist_coeffs2, H):
    x1, y1, x2, y2 = bboxes

    x1, x2 = x1 * ev_img_shape[0] / img_shape[0], x2 * ev_img_shape[0] / img_shape[0]
    y1, y2 = y1 * ev_img_shape[1] / img_shape[1] y2 * ev_img_shape[1] / img_shape[1]
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

    bbox = [x_ev_min, y_ev_min, x_ev_max, y_ev_max]
    return bbox
import numpy as np
import torch
import argparse
import time

from utils.read_events import read_events
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from utils.loading_utils import load_model, get_device
from utils.event_readers import FixedDurationEventReaderHDF5
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-e', '--event_file', required=True, type=str,
                        help='Path to the input event file')
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=True)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=20, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)
    parser.add_argument('--num_samples', default=50, type=int)
    parser.add_argument('--skip_samples', default=0, type=int)

    set_inference_options(parser)

    args = parser.parse_args()
    return args


def main(args):
    width, height = 1280, 720
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read events from hdf5 file
    events, triggers, id, ts = read_events(args.event_file)

    # Get events between the first and last trigger
    t_start = np.searchsorted(events['t'], triggers['t'][args.skip_samples])
    t_end = np.searchsorted(events['t'], triggers['t'][args.num_samples+args.skip_samples])
    idx_start = t_start
    idx_end = t_end

    for key in events.keys():
        events[key] = events[key][idx_start:idx_end]

    # Load the model
    model = load_model('pretrained/E2VID_lightweight.pth.tar')
    device = get_device(args.use_gpu)
    model.to(device)
    model.eval()

    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    if args.fixed_duration:
        event_window_iterator = FixedDurationEventReaderHDF5(events,
                                                         duration_ms=args.window_duration)

    start_index = 0

    with Timer('Processing entire dataset'):
        for event_window in event_window_iterator:

            last_timestamp = event_window[-1, 0]

            with Timer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else:
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)
            num_events_in_window = event_window.shape[0]
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

            start_index += num_events_in_window


if __name__ == '__main__':
    args = parse_args()
    main(args)
import h5py
import numpy as np


def read_events(file_path):
    events = dict()
    triggers = dict()

    with h5py.File(file_path, "r") as f:
        print("Starting to read triggers from file...")
        file_trg = f['EXT_TRIGGER']
        mask = np.asarray(file_trg['events']['p']) > 0
        for dset_str in ['p', 't']:
            triggers[dset_str] = np.asarray(file_trg['events']['{}'.format(dset_str)])[mask]

        print('Done!')
        del file_trg

        print("Starting to read events from file...")
        file_ev = f['CD']['events']
        for dset_str in ['x', 'y', 'p', 't']:
            events[dset_str] = np.asarray(file_ev['{}'.format(dset_str)])
            
        print('Done!')
        del file_ev

        file_idx = f['CD']['indexes']
        id = np.asarray(file_idx['id'])
        ts = np.asarray(file_idx['ts'])

    return events, triggers, id, ts
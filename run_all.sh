python3 reconstruct.py -i events/output.hdf5 --num_samples 100
python3 calibrate.py --num_samples 100
python3 reconstruct.py -i events/output.hdf5 --num_samples 100
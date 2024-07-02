num_samples_to_reconstruct=200
num_samples_to_calibrate=50
time_window=20

export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/hdf5/plugins 

python3 reconstruct.py -i events/output.hdf5 --num_samples $num_samples_to_reconstruct -T $time_window
python3 calibrate.py --num_samples $num_samples_to_calibrate
python3 visualise.py -i events/output.hdf5
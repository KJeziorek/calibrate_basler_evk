num_samples=500

export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/hdf5/plugins 

python3 reconstruct.py -i events/output.hdf5 --num_samples $num_samples
python3 calibrate.py --num_samples $num_samples
python3 reconstruct.py -i events/output.hdf5 --num_samples $num_samples
"""
Helper script to merge multiple collect play data into one hdf5 file.

Example usage:
    change the 'folder_path' to your robosuite installation

    python scripts/merge_playdata.py
"""

import h5py
import os

def copy_attributes(source, target):
    """Copy attributes from source to target"""
    for key, value in source.attrs.items():
        target.attrs[key] = value


# Path to your folder containing the hdf5 files
folder_path = "your_robosuite_path/robosuite/robosuite/models/assets/demonstrations/playdata"

# List all hdf5 files in the directory
hdf5_files = [f for f in os.listdir(folder_path) if f.endswith('.hdf5')]

counter = 0

# Create or open the merged.hdf5 file
with h5py.File(os.path.join(folder_path, "merged.hdf5"), "w") as merged_file:
    # Create a group named data if it doesn't exist yet
    data_group = merged_file.require_group("data")

    # Iterate over all the hdf5 files and merge demos
    for hdf5_file in hdf5_files:
        with h5py.File(os.path.join(folder_path, hdf5_file), 'r') as source_file:
            source_data_group = source_file['data']
            if counter == 0:
                copy_attributes(source_data_group, data_group)

            # Iterate through all demos in the 'data' group of the source file
            for demo_name in source_data_group:
                new_demo_name = f"demo_{counter}"

                # Copy demo to merged_file
                source_file.copy(f"data/{demo_name}", data_group, new_demo_name)

                # Copy attributes of the demo dataset
                copy_attributes(source_data_group[demo_name], data_group[new_demo_name])

                counter += 1

print("Merging completed!")
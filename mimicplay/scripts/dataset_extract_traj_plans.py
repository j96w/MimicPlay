"""
Helper script to extract 3D trajectory of the robot end-effector for the training of highlevel
latent planner in MimicPlay. Takes a dataset image_demo.hdf5 file as input.
It modifies the dataset in-place by adding a variable 'robot0_eef_pos_future_traj' in the observation.

Example usage:

    python scripts/dataset_extract_traj_plans.py --dataset 'datasets/playdata/image_demo.hdf5'
"""

import h5py
import numpy as np
import argparse

# Define constants
POINT_GAP = 15
FUTURE_POINTS_COUNT = 10

def get_future_points(arr):
    future_traj = []

    for i in range(POINT_GAP, (FUTURE_POINTS_COUNT + 1) * POINT_GAP, POINT_GAP):
        # Identify the indices for the current and prior points
        index_current = min(len(arr) - 1, i)

        current_point = arr[index_current]
        future_traj.extend(current_point)

    return future_traj

def process_dataset(dataset_file):
    # Open the HDF5 file in read+ mode (allows reading and writing)
    with h5py.File(dataset_file, 'r+') as f:
        demo_keys = [key for key in f['data'].keys() if 'demo_' in key]
        DEMO_COUNT = len(demo_keys)

        for i in range(0, DEMO_COUNT):
            # Extract the robot0_eef_pos data
            eef_pos = f[f'data/demo_{i}/obs/robot0_eef_pos'][...]

            # Calculate the future trajectory for each data point
            future_traj_data = np.array([get_future_points(eef_pos[j:]) for j in range(len(eef_pos))])

            # Create the new dataset
            f.create_dataset(f'data/demo_{i}/obs/robot0_eef_pos_future_traj', data=future_traj_data)

    print(f"Processed {DEMO_COUNT} demos!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process hdf5 dataset to generate future trajectory data.')
    parser.add_argument('--dataset', required=True, help='Path to the hdf5 dataset file.')

    args = parser.parse_args()
    process_dataset(args.dataset)
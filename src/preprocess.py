import os
import numpy as np

def read_skeleton_file(file_path):
    """
    Reads the NTU RGB+D skeleton file and returns array (num_frames, num_joints, 3).
    - Uses only the first body per frame (as common baseline practice).
    - Handles frames with zero bodies robustly (pads with zeros for such frames).
    """

    num_joints = 25  # NTU format uses 25 joints

    with open(file_path, 'r') as f:
        lines = f.readlines()

    n_frames = int(lines[0].strip())
    idx = 1
    all_frames = []
    for _ in range(n_frames):
        n_bodies = int(lines[idx].strip())
        idx += 1
        frame_skeleton = np.zeros((num_joints, 3), dtype=np.float32)
        found_body = False
        for b in range(n_bodies):
            body_info_start = idx
            idx += 1    # body ID info, skip
            n_joints = int(lines[idx].strip())
            idx += 1
            if not found_body:  # only use the first body sequence per frame
                for j in range(n_joints):
                    tokens = lines[idx].strip().split()
                    frame_skeleton[j, :] = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
                    idx += 1
                found_body = True
            else:
                idx += n_joints    # skip additional bodies' joints
        all_frames.append(frame_skeleton)

    return np.stack(all_frames, axis=0)  # shape (num_frames, num_joints, 3)


def load_all_skeletons(data_dir):
    skeletons = {}
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.skeleton'):
                file_path = os.path.join(root, file)
                skeletons[file] = read_skeleton_file(file_path)
    return skeletons

def normalize_skeleton(skeleton):
    root_coords = skeleton[:, 0:1, :]
    normalized = skeleton - root_coords
    return normalized

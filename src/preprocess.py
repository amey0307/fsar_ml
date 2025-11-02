import numpy as np

def read_skeleton_file(file_path):
    num_joints = 25  # NTU uses 25 joints
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
            idx += 1   # skip bodyID info
            n_joints = int(lines[idx].strip())
            idx += 1
            if not found_body:
                for j in range(n_joints):
                    tokens = lines[idx].strip().split()
                    frame_skeleton[j, :] = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
                    idx += 1
                found_body = True
            else:
                idx += n_joints
        all_frames.append(frame_skeleton)
    return np.stack(all_frames, axis=0)

def normalize_skeleton(skel_seq):
    # Mean subtraction (optional: normalize skeleton to root joint or pelvis)
    # Here: simple mean subtraction per joint axis, you can customize
    return skel_seq - skel_seq.mean(axis=(0, 1), keepdims=True)

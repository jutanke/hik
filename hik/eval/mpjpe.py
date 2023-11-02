import numpy as np
from typing import Dict


def mean_per_joint_l2_distance(a, b):
    assert a.shape == b.shape, "Input sequences must have the same shape"
    distances = np.sqrt(np.sum((a - b) ** 2, axis=2))
    mean_distances = np.mean(distances, axis=1)
    return mean_distances


def calc_mpjpe(results: Dict):
    """
    :param results: {
        "{action}": [
            {
                "Poses3d_in",
                "Masks_in",
                "Poses3d_out",
                "Masks_out",
                "frames_in",
                "Poses3d_out_pred",
                "target_pid",
                "pids"
            }
        ]
    }
    """
    pass
    # actions = results.keys()

    # ACTIONS = ["walking", "sitting_down", "whiteboard", "sink", "cupboard", "coffee"]

import unittest
import numpy as np
from hik.transforms.transforms import normalize, undo_normalization_to_seq
from typing import List, Tuple


def normalize3d(Poses3d, frame: int):
    """
    :param Poses3d: {n_frames x n_person x 29 x 3}
    """
    if len(Poses3d.shape) != 4:
        raise ValueError(f"(1) Weird shape: {Poses3d.shape}")
    if Poses3d.shape[2] != 29 or Poses3d.shape[3] != 3:
        raise ValueError(f"(2) Weird shape: {Poses3d.shape}")
    Poses3d = np.copy(Poses3d)
    Norm_params = []
    n_person = Poses3d.shape[1]
    if frame == -1:
        frame = len(Poses3d) - 1
    for pid in range(n_person):
        seq = Poses3d[:, pid]
        seq_norm, norm_params = normalize(
            seq, frame=frame, return_transform=True
        )  # noqa E501
        Poses3d[:, pid] = seq_norm
        Norm_params.append(norm_params)
    return Poses3d, Norm_params


def denormalize3d(
    Poses3d: np.ndarray, Norm_params: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    :param Poses3d: {n_frames x n_person x 29 x 3}
    """
    if len(Poses3d.shape) != 4:
        raise ValueError(f"(1) Weird shape: {Poses3d.shape}")
    if Poses3d.shape[2] != 29 or Poses3d.shape[3] != 3:
        raise ValueError(f"(2) Weird shape: {Poses3d.shape}")

    Poses3d = np.copy(Poses3d)
    n_person = Poses3d.shape[1]

    if len(Norm_params) != n_person:
        raise ValueError(
            f"we have {n_person} persons but {len(Norm_params)} norm params.."
        )

    for pid in range(n_person):
        mu, R = Norm_params[pid]
        seq = Poses3d[:, pid]
        seq = undo_normalization_to_seq(seq, mu, R)
        Poses3d[:, pid] = seq
    return Poses3d


def single_backfill_masked(poses3d, mask):
    """
    :param poses3d: {n_frames x 29 x 3}
    :param mask: {n_frames}
    """
    if len(poses3d.shape) != 3:
        raise ValueError(f"(1) Weird shape: {poses3d.shape}")
    if len(mask.shape) != 1:
        raise ValueError(f"(2) Weird shape: {mask.shape}")
    if mask.shape[0] != poses3d.shape[0]:
        raise ValueError(f"(3) Weird shape: {mask.shape} vs {poses3d.shape}")
    if poses3d.shape[1] != 29 or poses3d.shape[2] != 3:
        raise ValueError(f"(4) Weird shape: {poses3d.shape}")
    poses3d = np.copy(poses3d)

    n_frames = poses3d.shape[0]
    valid_pose = None
    for t in range(n_frames - 1, -1, -1):
        if mask[t] > 0.5:
            valid_pose = poses3d[t]

        if mask[t] < 0.5 and valid_pose is not None:
            poses3d[t] = valid_pose
            mask[t] = 0.51

    return poses3d, mask


def backfill_masked(Poses3d, Mask):
    """
    :param Poses3d: {n_frames x n_person x 29 x 3}
    :param Mask: {n_frames x n_person}
    """
    if len(Poses3d.shape) != 4:
        raise ValueError(f"(1) Weird shape: {Poses3d.shape}")
    if len(Mask.shape) != 2:
        raise ValueError(f"(2) Weird shape: {Mask.shape}")
    if Mask.shape[0] != Poses3d.shape[0] or Mask.shape[1] != Poses3d.shape[1]:
        raise ValueError(f"(3) Weird shape: {Mask.shape} vs {Poses3d.shape}")
    if Poses3d.shape[2] != 29 or Poses3d.shape[3] != 3:
        raise ValueError(f"(4) Weird shape: {Poses3d.shape}")
    Poses3d = np.copy(Poses3d)

    n_frames = Poses3d.shape[0]
    n_person = Poses3d.shape[1]
    for pid in range(n_person):
        valid_pose = None
        for t in range(n_frames - 1, -1, -1):
            if Mask[t, pid] > 0.5:
                valid_pose = Poses3d[t, pid]

            if Mask[t, pid] < 0.5 and valid_pose is not None:
                Poses3d[t, pid] = valid_pose
                Mask[t, pid] = 0.51
    return Poses3d, Mask


# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# u n i t t e s t
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


class UtilsTestCase(unittest.TestCase):
    def test_normdenorm(self):
        n_frames = 10
        n_person = 2
        Poses3d = np.random.random(size=(n_frames, n_person, 29, 3)) * 10

        Poses3d_norm, norm_data = normalize3d(Poses3d=Poses3d, frame=-1)
        Poses3d_rec = denormalize3d(Poses3d_norm, norm_data)

        self.assertEqual(len(norm_data), n_person)
        diff = np.max(np.abs(Poses3d_rec - Poses3d))
        self.assertAlmostEqual(diff, 0.0, places=4)

    def test_backfill_masked_allzero(self):
        n_frames = 10
        n_person = 1
        Persons3d = np.zeros((n_frames, n_person, 29, 3))
        Masks = np.ones((n_frames, n_person))
        _, Masks_ = backfill_masked(Poses3d=Persons3d, Mask=Masks)
        self.assertTrue(np.min(Masks_) > 0.1)

    def test_backfill_masked_allone(self):
        n_frames = 10
        n_person = 1
        Persons3d = np.zeros((n_frames, n_person, 29, 3))
        for i in range(n_frames):
            Persons3d[i, 0] = i + 1
        Masks = np.zeros((n_frames, n_person))
        Persons3d_, Masks_ = backfill_masked(Poses3d=Persons3d, Mask=Masks)
        self.assertTrue(np.min(Masks_) < 0.1)
        diff = np.abs(Persons3d_ - Persons3d)
        self.assertTrue(np.max(diff) < 0.000001)

    def test_backfill_masked(self):
        n_frames = 10
        n_person = 1
        Persons3d = np.zeros((n_frames, n_person, 29, 3))
        for i in range(n_frames):
            Persons3d[i, 0] = i + 1
        Persons3d_goal = np.copy(Persons3d)
        Persons3d_goal[[0, 1, 2]] = Persons3d_goal[3]

        Masks = np.ones((n_frames, n_person))
        Masks[:3] = 0
        Persons3d_, Masks_ = backfill_masked(Poses3d=Persons3d, Mask=Masks)

        self.assertTrue(np.min(Masks_) < 0.511)
        diff = np.abs(Persons3d_ - Persons3d_goal)
        self.assertTrue(np.max(diff) < 0.000001)

    def test_backfill_masked_v2(self):
        n_frames = 10
        n_person = 1
        Persons3d = np.zeros((n_frames, n_person, 29, 3))
        for i in range(n_frames):
            Persons3d[i, 0] = i + 1
        Persons3d_goal = np.copy(Persons3d)
        Persons3d_goal[[0, 1, 2]] = Persons3d_goal[3]

        Masks = np.ones((n_frames, n_person))
        Masks[:3] = 0
        Masks[8:] = 0
        Persons3d_, Masks_ = backfill_masked(Poses3d=Persons3d, Mask=Masks)

        self.assertTrue(np.min(Masks_) < 0.1)
        diff = np.abs(Persons3d_ - Persons3d_goal)
        self.assertTrue(np.max(diff) < 0.000001)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import numpy.linalg as la
import hik.transforms.rotation as rot

from einops import repeat


def __assert_seq_shape(seq):
    """
    :param seq: {n_frames x 29 x 3}
    """
    if len(seq.shape) != 3 or not (seq.shape[1] == 29 and seq.shape[2] == 3):
        raise ValueError(f"Weird seq shape: {seq.shape}")


def normalize_smpl(
    seq3d,
    betas,
    frame: int,
    jid_left=1,
    jid_right=2,
    allow_zero_z=False,
):
    """
    normalize the 3d sequence and get the proper SMPL transform
    :param seq: {n_frames x 29 x 3}
    :param betas: {n_frames x 10} OR {10}
    :param frame: {int}
    """
    __assert_seq_shape(seq3d)
    if len(betas.shape) == 1:
        if betas.shape[0] != 10:
            raise ValueError(f"(1) Weird beta shape: {betas.shape}")
        n_frames = seq3d.shape[0]
        betas = repeat(betas, "d -> n d", n=n_frames)
    elif len(betas.shape) == 2:
        if betas.shape[1] != 10 or betas.shape[0] != seq3d.shape[0]:
            raise ValueError(f"(2) Weird beta shape: {betas.shape}, seq: {seq3d.shape}")
    else:
        raise ValueError(f"(3) Weird beta shape: {betas.shape}")

    seq_norm = normalize(
        seq=seq3d,
        frame=frame,
        jid_left=jid_left,
        jid_right=jid_right,
        allow_zero_z=allow_zero_z,
    )

    # TODO cont' here


def normalize(
    seq,
    frame: int,
    jid_left=1,
    jid_right=2,
    return_transform=False,
    allow_zero_z=False,
    zero_z=True,
    check_shape=True,
):
    """
    :param seq: {n_frames x 29 x 3}
    :param frame: {int}
    """
    if check_shape:
        __assert_seq_shape(seq)
    else:
        assert len(seq.shape) == 3 and seq.shape[2] == 3
    left3d = seq[frame, jid_left]
    right3d = seq[frame, jid_right]

    if not allow_zero_z:
        if np.isclose(left3d[2], 0.0):
            raise ValueError(f"Left seems to be zero! -> {left3d}")
        if np.isclose(right3d[2], 0.0):
            raise ValueError(f"Right seems to be zero! -> {right3d}")

    mu, R = get_normalization(left3d, right3d, zero_z=zero_z)
    if return_transform:
        return apply_normalization_to_seq(
            seq, mu, R, check_shape=check_shape
        ), (  # noqa E501
            mu,
            R,
        )  # noqa E501
    else:
        return apply_normalization_to_seq(seq, mu, R, check_shape=check_shape)


def get_normalization(left3d, right3d, zero_z=True):
    """
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    if zero_z:
        mu[2] = 0
    left2d = left3d[:2]
    right2d = right3d[:2]
    y = right2d - left2d
    y = y / (la.norm(y) + 0.00000001)
    angle = np.arctan2(y[1], y[0])
    R = rot.rot3d(0, 0, angle)
    return mu, R


def undo_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 29 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    __assert_seq_shape(seq)
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    R_T = np.transpose(R)
    seq = rot.apply_rotation_to_seq(seq, R_T)
    seq = seq + mu
    return seq


def apply_normalization_to_points3d(pts3d, mu, R):
    """
    :param pts3d: {n_points x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    mu = np.expand_dims(np.squeeze(mu), axis=0)
    pts3d = pts3d - mu
    return np.ascontiguousarray(pts3d @ R)


def apply_normalization_to_seq(seq, mu, R, check_shape=True):
    """
    :param seq: {n_frames x 29 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    if check_shape:
        __assert_seq_shape(seq)
    else:
        assert len(seq.shape) == 3 and seq.shape[2] == 3
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    seq = seq - mu
    return rot.apply_rotation_to_seq(seq, R)


# =======================================
# U N I T  T E S T S
# =======================================


class TestTransforms(unittest.TestCase):
    def test_normalize_redo(self):
        poses = np.random.random((10, 29, 3))
        poses_norm, (mu, R) = normalize(poses, frame=5, return_transform=True)
        poses_undo = undo_normalization_to_seq(poses_norm, mu=mu, R=R)
        dist = np.abs(poses_undo - poses)
        self.assertAlmostEqual(np.max(dist), 0.0)


if __name__ == "__main__":
    unittest.main()

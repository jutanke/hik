import numpy as np
from scipy.spatial.transform import Rotation as R
from einops import rearrange


def rot3d(a, b, c):
    """"""
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]],  # noqa E501
        np.float32,
    )
    Ry = np.array(
        [[np.cos(b), 0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0, np.cos(b)]],  # noqa E501
        np.float32,
    )
    Rz = np.array(
        [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0.0, 0.0, 1.0]],  # noqa E501
        np.float32,
    )
    return np.ascontiguousarray(Rx @ Ry @ Rz)


def quaternion2rvecs(quats, mask: np.ndarray = None):
    """
    :param quats: {n x 4} or {n x 21 x 4}
    :param mask: {n} SOME of the quaternions might be zero!
    """
    if quats.shape[-1] != 4:
        raise ValueError(
            f"(1) Incorrect quaternion shape: {quats.shape}, expect (n x 4) or (n x 21 x 4)")  # noqa E501
    if len(quats.shape) == 2:
        if mask is not None and np.max(mask) < 0.5:
            return np.zeros((len(quats), 3), dtype=np.float32)
        elif mask is None or np.min(mask) > 0.5:
            return R.from_quat(quats).as_rotvec().astype("float32")
        else:
            if len(mask.shape) > 1 or len(mask) != len(quats):
                raise ValueError(
                    f"(1) Mask and rotations don't fit: mask:{mask.shape} vs {quats.shape}")  # noqa E501
            rvecs = []
            zero = np.zeros((3,), dtype=np.float32)
            for quat, m in zip(quats, mask):
                if m > 0.5:
                    rvecs.append(R.from_quat(quat).as_rotvec())
                else:
                    rvecs.append(zero)
            return np.array(rvecs, dtype=np.float32)
    elif len(quats.shape) == 3:
        if quats.shape[1] != 21:
            raise ValueError(
                f"(2) Incorrect quaternion shape: {quats.shape}, expect (n x 4) or (n x 21 x 4)")  # noqa E501
        if mask is not None and np.max(mask) < 0.5:
            return np.zeros((len(quats), 21, 3), dtype=np.float32)
        elif mask is None or np.min(mask) > 0.5:
            quats = rearrange(quats, "t j d -> (t j) d")
            return rearrange(R.from_quat(quats).as_rotvec(),
                             "(t j) d -> t j d", j=21).astype('float32')
        else:
            if len(mask.shape) > 1 or len(mask) != len(quats):
                raise ValueError(
                    f"(2) Mask and rotations don't fit: mask:{mask.shape} vs {quats.shape}")  # noqa E501

            indices = np.nonzero(mask)[0]
            quats_select = rearrange(quats[indices], "t j d -> (t j) d")
            rvecs_select = R.from_quat(
                quats_select).as_rotvec().astype('float32')
            rvecs_select = rearrange(rvecs_select,
                                     "(t j) d -> t j d", j=21
                                     )
            rvecs = np.zeros((len(mask), 21, 3), dtype=np.float32)
            rvecs[indices] = rvecs_select
            return rvecs
    else:
        raise ValueError(
            f"(3) Incorrect quaternion shape: {quats.shape}, expect (n x 4) or (n x 21 x 4)")  # noqa E501


def rvecs2quaternion(rvecs):
    """
    :param rvecs: {n x 3}
    """
    quats = []
    for rvec in rvecs:
        quat = R.from_rotvec(rvec).as_quat()
        if len(quats) > 0:
            if quat @ quats[-1] < 0:
                # antipodal quaterions represent the
                # same rotation: ensure that the seq
                # is on the same hemisphere to prevent
                # the networks from having to learn crazy
                # jumps
                quat = quat * -1
        quats.append(quat)
    return np.array(quats, dtype=np.float32)


def apply_rotation_to_seq(seq, R):
    """
    :param seq: {n_frames x 29 x 3}
    :param R: {3x3}
    """
    R = np.expand_dims(R, axis=0)
    return np.ascontiguousarray(seq @ R)

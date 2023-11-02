from smplx.body_models import SMPLX
from os.path import join
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from scipy.spatial.transform import Rotation
import numpy as np


def transform_pose(pose3d, t, R):
    """
    :param pose3d: {24 x 3}
    """
    return np.transpose(R.dot(np.transpose(pose3d))) + t


def find_transformation(src, tgt):
    """
    n = 3
    :param src: {n x 3}
    :param tgt: {n x 3}
    """
    src_center = np.mean(src, axis=0)
    tgt_center = np.mean(tgt, axis=0)
    src_zero_centered = src - src_center
    tgt_zero_centered = tgt - tgt_center
    covariance_matrix = np.dot(tgt_zero_centered.T, src_zero_centered)
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Ensure a right-handed coordinate system (Determinant should be positive)
    if np.linalg.det(np.dot(U, Vt)) < 0:
        Vt[2, :] *= -1
    R = np.dot(U, Vt)
    t = tgt_center - np.dot(R, src_center)
    return t, R


class Body:
    def __init__(self, smplx_path: str) -> None:
        bm_path = join(smplx_path, "SMPLX_NEUTRAL.npz")
        self.bm = SMPLX(bm_path, use_pca=False)

    def batch_transform_to_smpl_canonical_space(self, betas, poses3d):
        """
        :param betas: {10,}
        :param poses3d: {n_frames x 24 x 3}
        """
        if (
            len(poses3d) != 3
            or poses3d.shape[2] != 3
            or poses3d.shape[1] != 24  # noqa E501
        ):
            raise ValueError(f"Weird shape: {poses3d.shape}")

        canonical_pose3d = self.get_canonical_pose(betas=betas)
        return [
            transform_pose(
                pose, *find_transformation(pose[:3], canonical_pose3d[:3])
            )  # noq E501
            for pose in poses3d
        ]

    def get_canonical_pose(self, betas, return_vertices=False):
        translation = np.zeros((3,), dtype=np.float32)
        rotation = np.zeros((3,), dtype=np.float32)
        pose = np.zeros((21, 3), dtype=np.float32)
        return self.render(
            betas=betas,
            pose=pose,
            translation=translation,
            rotation=rotation,
            return_vertices=return_vertices,
        )

    def get_global_transformation(self, betas, pose3d):
        """
        :param betas
        :param pose3d: {24 x 3}
        """
        canonical_pose3d = self.get_canonical_pose(betas)
        return find_transformation(
            src_pts=pose3d[:3], tgt_pts=canonical_pose3d[:3]
        )  # noqa E501

    @torch.no_grad()
    def render_batch(
        self,
        betas,
        pose,
        translation,
        rotation,
        return_head=True,
        use_tqdm=False,
    ):  # noqa E501
        """
        :param betas: {n_batch x 10}
        :param pose: {n_batch x 21 x 3}
        :param translation: {n_batch x 3}
        :param rotation: {n_batch x 3}
        """
        RIGHT_EAR_ID = 4
        RIGHT_EYE_ID = 1320
        LEFT_EYE_ID = 2595
        NOSE_ID = 2798
        LEFT_EAR_ID = 3020

        device = torch.device("cpu")
        bm = self.bm.to(device)

        betas = torch.from_numpy(betas)
        body_pose = torch.from_numpy(pose)
        translation = torch.from_numpy(translation)
        rotation = torch.from_numpy(rotation)

        # betas = rearrange(betas, "d -> 1 d")

        n_batch = len(body_pose)

        jaw_pose = repeat(bm.jaw_pose, "a d -> (a b) d", b=n_batch)
        reye_pose = repeat(bm.reye_pose, "a d -> (a b) d", b=n_batch)
        leye_pose = repeat(bm.leye_pose, "a d -> (a b) d", b=n_batch)
        right_hand_pose = repeat(bm.right_hand_pose, "a d -> (a b) d", b=n_batch)
        left_hand_pose = repeat(bm.left_hand_pose, "a d -> (a b) d", b=n_batch)
        expression = repeat(bm.expression, "a d -> (a b) d", b=n_batch)

        dataset = SMPLInputDataset(
            betas=betas,
            body_pose=body_pose,
            translation=translation,
            rotation=rotation,
            jaw_pose=jaw_pose,
            reye_pose=reye_pose,
            leye_pose=leye_pose,
            right_hand_pose=right_hand_pose,
            left_hand_pose=left_hand_pose,
            expression=expression,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=2048)
        Js = []

        for batch in tqdm(
            dataloader,
            leave=True,
            position=0,
            total=len(dataloader),
            disable=not use_tqdm,
        ):
            out = bm(
                betas=batch["betas"],
                body_pose=batch["body_pose"],
                transl=batch["translation"],
                global_orient=batch["rotation"],
                jaw_pose=batch["jaw_pose"],
                reye_pose=batch["reye_pose"],
                leye_pose=batch["leye_pose"],
                right_hand_pose=batch["right_hand_pose"],
                left_hand_pose=batch["left_hand_pose"],
                expression=batch["expression"],
                return_verts=True,
            )

            J = out.joints[:, :24].cpu().numpy().copy()

            if return_head:
                V = out.vertices[:].cpu().numpy().copy()
                F = V[
                    :,
                    [
                        NOSE_ID,
                        LEFT_EYE_ID,
                        RIGHT_EYE_ID,
                        LEFT_EAR_ID,
                        RIGHT_EAR_ID,
                    ],  # noqa E501
                ]  # noqa E501
                J = np.concatenate([J, F], axis=1)

            Js.append(J)

        Js = np.concatenate(Js, axis=0)
        return Js

    @torch.no_grad()
    def render(
        self,
        betas,
        pose,
        translation,
        rotation,
        return_vertices=False,
        return_head=True,
    ):  # noqa E501
        RIGHT_EAR_ID = 4
        RIGHT_EYE_ID = 1320
        LEFT_EYE_ID = 2595
        NOSE_ID = 2798
        LEFT_EAR_ID = 3020

        device = torch.device("cpu")
        bm = self.bm.to(device)

        betas = torch.from_numpy(betas)
        body_pose = torch.from_numpy(pose)
        translation = torch.from_numpy(translation)
        rotation = torch.from_numpy(rotation)

        betas = rearrange(betas, "d -> 1 d")
        translation = rearrange(translation, "d -> 1 d")
        rotation = rearrange(rotation, "d -> 1 d")
        body_pose = rearrange(body_pose, "j d -> 1 j d")

        out = bm(
            betas=betas,
            body_pose=body_pose,
            transl=translation,
            global_orient=rotation,
            return_verts=True,
        )

        J = out.joints[:, :24].cpu().numpy().copy()

        if return_vertices:
            V = out.vertices[0].cpu().numpy().copy()
            return J[0], V

        if return_head:
            V = out.vertices[0].cpu().numpy().copy()
            F = V[
                [NOSE_ID, LEFT_EYE_ID, RIGHT_EYE_ID, LEFT_EAR_ID, RIGHT_EAR_ID]
            ]  # noqa E501
            return np.concatenate([J[0], F], axis=0)

        return J[0]


class SMPLInputDataset(Dataset):
    def __init__(
        self,
        betas,
        body_pose,
        translation,
        rotation,
        jaw_pose,
        reye_pose,
        leye_pose,
        right_hand_pose,
        left_hand_pose,
        expression,
    ):
        super().__init__()

        self.betas = betas
        self.body_pose = body_pose
        self.translation = translation
        self.rotation = rotation
        self.jaw_pose = jaw_pose
        self.reye_pose = reye_pose
        self.leye_pose = leye_pose
        self.right_hand_pose = right_hand_pose
        self.left_hand_pose = left_hand_pose
        self.expression = expression

        self.n_entries = len(self.body_pose)

    def __len__(self):
        return self.n_entries

    def __getitem__(self, index):
        return {
            "betas": self.betas[index],
            "body_pose": self.body_pose[index],
            "translation": self.translation[index],
            "rotation": self.rotation[index],
            "jaw_pose": self.jaw_pose[index],
            "reye_pose": self.reye_pose[index],
            "leye_pose": self.leye_pose[index],
            "right_hand_pose": self.right_hand_pose[index],
            "left_hand_pose": self.left_hand_pose[index],
            "expression": self.expression[index],
        }

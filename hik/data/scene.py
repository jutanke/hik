import numpy as np
from hik.data import PersonSequences
from hik.data.kitchen import Kitchen
from hik.data.constants import pids_per_dataset, activity2index, LAST_FRAMES
from hik.data.utils import get_splits
from hik.data.smpl import Body
import hik.transforms.rotation as rot
from tqdm import tqdm
from einops import rearrange, repeat


class Scene:
    @staticmethod
    def load_from_paths(
        dataset: str,
        person_path: str,
        scene_path: str,
        smplx_path: str,
        *,
        as_quaternions: bool = False,
    ):
        """
        :param dataset: the selected dataset {A, B, C, D}
        :param person_path: path to persons
        :param scene_path: path to scenes
        """
        kitchen = Kitchen.load_for_dataset(dataset=dataset, data_location=scene_path)
        person_seqs = PersonSequences(person_path=person_path)
        return Scene(
            dataset=dataset,
            person_seqs=person_seqs,
            kitchen=kitchen,
            as_quaternions=as_quaternions,
            smplx_path=smplx_path,
        )

    def __init__(
        self,
        dataset: str,
        person_seqs: PersonSequences,
        kitchen: Kitchen,
        smplx_path: str,
        *,
        as_quaternions: bool = False,
        use_tqdm=False,
    ):
        """
        :param dataset: {A, B, C, D}
        """
        self.dataset = dataset
        self.body = Body(smplx_path=smplx_path)
        self.as_quaternions = as_quaternions
        self.person_seqs = person_seqs
        self.kitchen = kitchen
        self.pid2index = {}
        for index, pid in enumerate(pids_per_dataset[kitchen.dataset]):
            self.pid2index[pid] = index

        n_frames = LAST_FRAMES[kitchen.dataset]
        n_persons = len(pids_per_dataset[kitchen.dataset])
        n_activities = len(activity2index)
        rot_dim = 4 if as_quaternions else 3

        self.poses3d = np.zeros((n_frames, n_persons, 29, 3), dtype=np.float32)
        self.masks = np.zeros((n_frames, n_persons), dtype=np.float32)
        self.activities = np.zeros(
            (n_frames, n_persons, n_activities), dtype=np.float32
        )
        self.transforms = np.zeros((n_frames, n_persons, 3 + rot_dim), dtype=np.float32)
        self.smpls = np.zeros((n_frames, n_persons, 21, rot_dim), dtype=np.float32)
        self.betas = np.zeros((n_persons, 10), dtype=np.float32)

        frames_set = set()
        for frame in tqdm(
            range(1, n_frames),
            leave=True,
            total=n_frames,
            position=0,
            disable=not use_tqdm,
        ):
            for person_dict in self.person_seqs.get_frame(
                dataset=self.kitchen.dataset,
                frame=frame,
                as_quaternions=as_quaternions,
            ):
                betas = person_dict["betas"]
                pid = person_dict["pid"]
                frames_set.add(frame)
                index = self.pid2index[pid]
                self.betas[index] = betas
                self.poses3d[frame, index] = person_dict["pose3d"]
                self.masks[frame, index] = 1.0
                self.activities[frame, index] = person_dict["act"]
                self.transforms[frame, index] = person_dict["transforms"]
                self.smpls[frame, index] = person_dict["smpl"]
        self.frames = list(sorted(frames_set))

    def get_splits(self, length: int, stepsize: int = -1):
        """ """
        stepsize = length if stepsize == -1 else stepsize

        splits = get_splits(self.frames, length=length, stepsize=stepsize)

        poses3d = []
        smpls = []
        transforms = []
        masks = []
        acts = []
        start_frames = []

        for start_frame in splits:
            start_frames.append(start_frame)
            poses3d.append(self.poses3d[start_frame : start_frame + length].copy())
            smpls.append(self.smpls[start_frame : start_frame + length].copy())
            transforms.append(
                self.transforms[start_frame : start_frame + length].copy()
            )
            masks.append(self.masks[start_frame : start_frame + length].copy())
            acts.append(self.activities[start_frame : start_frame + length].copy())

        acts = np.array(acts, dtype=np.float32)
        poses3d = np.array(poses3d, dtype=np.float32)
        smpls = np.array(smpls, dtype=np.float32)
        transforms = np.array(transforms, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        start_frames = np.array(start_frames, dtype=np.int64)

        return {
            "poses3d": poses3d,  # n_seq x n_frames x n_person x 29 x 3
            "smpls": smpls,  # n_seq x n_frames x n_person x 21 x 3|4
            "transforms": transforms,  # n_seq x n_frames x n_person x 3+(3|4)
            "masks": masks,  # n_seq x n_frames n_person
            "activities": acts,  # n_seq x n_frames x  n_person x 82
            "start_frames": start_frames,  # n_seq
        }

    def data_quat2rvec(self, data, rerender3d=False, use_tqdm=False):
        """
        :param data
        """
        smpls_quat = data["smpls"]
        if smpls_quat.shape[-1] != 4:
            raise ValueError(
                f"This seems to not be a quaternion data rep: {smpls_quat.shape}"
            )  # noqa E501
        masks = data["masks"]
        s, t, p = masks.shape

        masks_flat = rearrange(masks, "s t p -> (s t p)")
        smpls_quat_flat = rearrange(smpls_quat, "s t p j d -> (s t p) j d")

        smpls_rvec_flat = rot.quaternion2rvecs(smpls_quat_flat, mask=masks_flat)
        smpls_rvec = rearrange(
            smpls_rvec_flat, "(s t p) j d -> s t p j d", p=p, s=s, t=t
        )

        transforms = data["transforms"][:, :, :, :6].copy()
        transforms_rot_quat_flat = rearrange(
            data["transforms"][:, :, :, 3:], "s t p d -> (s t p) d"
        )
        transforms_rvec_flat = rot.quaternion2rvecs(
            transforms_rot_quat_flat, mask=masks_flat
        )
        transforms_rvec = rearrange(
            transforms_rvec_flat, "(s t p) d -> s t p d", s=s, t=t, p=p
        )
        transforms[:, :, :, 3:] = transforms_rvec

        # TODO: render/rasterize 3d positions
        # smpls_rvec -> {batch x n_frames x n_person x 21 x 3}
        smpls_rvec_flat = rearrange(smpls_rvec, "b t p j d -> (b t p) j d")
        indices = np.nonzero(masks_flat)[0]

        smpls_rvec_flat_selected = smpls_rvec_flat[indices]
        b = transforms.shape[0]
        t = transforms.shape[1]
        transforms_flat = rearrange(transforms, "b t p d -> (b t p) d")
        transforms_flat_selected = transforms_flat[indices]
        betas_flat = repeat(self.betas, "p d -> (b t p) d", b=b, t=t)
        betas_flat_select = betas_flat[indices]

        poses3d = np.zeros_like(data["poses3d"])  # b t p j d
        if rerender3d:
            b = poses3d.shape[0]
            t = poses3d.shape[1]
            p = poses3d.shape[2]
            poses3d_flat = rearrange(poses3d, "b t p j d -> (b t p) j d")

            Js_flat = self.body.render_batch(
                betas=betas_flat_select,
                pose=smpls_rvec_flat_selected,
                translation=transforms_flat_selected[:, :3],
                rotation=transforms_flat_selected[:, 3:],
                use_tqdm=use_tqdm,
            )
            poses3d_flat[indices] = Js_flat
            poses3d = rearrange(poses3d_flat, "(b t p) j d -> b t p j d", b=b, t=t, p=p)

        return {
            "poses3d": poses3d,
            "smpls": smpls_rvec,
            "transforms": transforms,
            "masks": masks,
            "activities": data["activities"].copy(),
        }

import numpy as np
from os import listdir
from os.path import join, isfile
from tqdm import tqdm
import hik.transforms.rotation as rot
from hik.data.constants import activity2index, pids_per_dataset
from einops import rearrange
import tempfile
import hashlib

from typing import List


class PersonSequence:
    def __init__(self, data, dataset: str, pid: int):
        """
        :param data: {
            "transforms": {n x 6}
            "smpl": {n x 21 x 3}
            "poses3d": {n x 29 x 3}
            "frames": {n}
            "act": {n x 82}
            "betas": {10}
        }
        """
        self.dataset = dataset
        self.pid = pid
        self.transforms = data["transforms"].astype("float32")
        self.smpl = data["smpl"].astype("float32")
        self.poses3d = data["poses3d"].astype("float32")
        self.frames = data["frames"]
        self.act = data["act"].astype("float32")
        self.betas = data["betas"].astype("float32")

        if len(self.frames) != len(self.poses3d):
            raise ValueError("Inconsistent lengths: frames vs poses3d")

        if len(self.act) != len(self.poses3d):
            raise ValueError("Inconsistent lengths: act vs poses3d")

        if len(self.smpl) != len(self.poses3d):
            raise ValueError("Inconsistent lengths: smpl vs poses3d")

        if len(self.transforms) != len(self.poses3d):
            raise ValueError("Inconsistent lengths: transforms vs poses3d")

        self.transforms_as_quat = None
        self.smpl_as_quat = None

        smpl_quat_tmp_fname, trans_quat_tmp_fname = self._get_quat_tempfiles()
        if isfile(smpl_quat_tmp_fname):
            self.smpl_as_quat = np.load(smpl_quat_tmp_fname)
        if isfile(trans_quat_tmp_fname):
            self.transforms_as_quat = np.load(trans_quat_tmp_fname)

        self.frame2index = {}
        self.index2frame = {}
        for i, t in enumerate(self.frames):
            self.frame2index[t] = i
            self.index2frame[i] = t

        # calculate uid
        loc3d = rearrange(
            self.transforms[:, :3].astype("int64"), "t d -> (t d)"
        )  # noqa E501
        betas_hard = (self.betas * 100).astype("int64")
        uid_str = f"{dataset}_{pid}_{len(loc3d)}_{loc3d}_betas{betas_hard}"
        uid = hashlib.sha1(uid_str.encode("utf-8")).hexdigest()
        self.uid = f"{dataset}_{pid}_{uid}"

    def is_valid_at_frame(self, frame: int):
        return frame in self.frame2index

    def __len__(self):
        return len(self.poses3d)

    def __getitem__(self, index):
        frame = self.index2frame[index]
        return self.get_frame(frame=frame)

    def get_frames_where_action(self, actions: List[str]) -> List[int]:
        """
        find all frames where the actions are happening
        """
        frames = []
        for action in actions:
            if action not in activity2index:
                raise ValueError(f"{action} not a valid action name!")

            index = activity2index[action]
            act = (self.act[:, index] > 0.5) * 1
            if np.max(act) > 0.5:
                indices = np.nonzero(act)[0]
                frames.append(self.frames[indices])

        if len(frames) > 0:
            frames = np.concatenate(frames, axis=0)
            frames = list(sorted(set(frames)))

        return frames

    def get_uid(self):
        return self.uid

    def _get_quat_tempfiles(self):
        min_frame = min(self.frames)
        max_frame = max(self.frames)
        num_frames = len(self.frames)
        return join(
            tempfile.gettempdir(),
            f"smpl_ds{self.dataset}_pid{self.pid}_frames{num_frames}_{min_frame}_{max_frame}.npy",  # noqa E501
        ), join(
            tempfile.gettempdir(),
            f"transforms_ds{self.dataset}_pid{self.pid}_frames{num_frames}_{min_frame}_{max_frame}.npy",  # noqa E501
        )

    def get_transforms_as_quaternion(self):
        """
        converts rvec transform params to quaternion params
        """
        if self.transforms_as_quat is None:
            _, trans_quat_tmp_fname = self._get_quat_tempfiles()
            ts = self.transforms[:, :3]
            rvecs = self.transforms[:, 3:]
            quats = rot.rvecs2quaternion(rvecs=rvecs)
            self.transforms_as_quat = np.concatenate([ts, quats], axis=1)
            np.save(trans_quat_tmp_fname, self.transforms_as_quat)
        return self.transforms_as_quat

    def get_smpl_as_quaterions(self):
        """
        converts rvec smpl params into quaternion params
        """
        if self.smpl_as_quat is None:
            smpl_quat_tmp_fname, _ = self._get_quat_tempfiles()
            smpl_as_quat = []
            for jid in range(21):
                smpl_as_quat.append(
                    rot.rvecs2quaternion(rvecs=self.smpl[:, jid])
                )  # noqa E501
            # jid x t x 3
            smpl_as_quat = np.array(smpl_as_quat, dtype=np.float32)
            self.smpl_as_quat = rearrange(smpl_as_quat, "j t d -> t j d")
            np.save(smpl_quat_tmp_fname, self.smpl_as_quat)
        return self.smpl_as_quat

    def get_range3d(self, start_frame: int, end_frame: int):
        """
        get the range
        """
        if start_frame >= end_frame:
            raise ValueError(f"conflict: {start_frame} < {end_frame}")

        n_frames = end_frame - start_frame
        mask = np.zeros((n_frames), dtype=np.float32)
        poses3d = np.zeros((n_frames, 29, 3), dtype=np.float32)
        for i, frame in enumerate(range(start_frame, end_frame)):
            if frame in self.frame2index:
                index = self.frame2index[frame]
                pose3d = self.poses3d[index]
                mask[i] = 1.0
                poses3d[i] = pose3d
        return poses3d, mask

    def get_frame(self, frame, as_quaternions=False):
        index = self.frame2index[frame]
        if as_quaternions:
            transforms = self.get_transforms_as_quaternion()
            smpl = self.get_smpl_as_quaterions()
        else:
            transforms = self.transforms
            smpl = self.smpl
        return {
            "pose3d": self.poses3d[index],
            "act": self.act[index],
            "smpl": smpl[index],
            "transforms": transforms[index],
            "betas": self.betas,
            "pid": self.pid,
            "frame": frame,
        }


class PersonSequences:
    def __init__(self, person_path: str):
        self.dataset_pid_lookup = {}  # (dataset, pid) -> [..]
        self.dataset_frame_lookup = {}
        self.dataset_lookup = {}

        for fname in tqdm(
            [f for f in listdir(person_path) if f.endswith(".npz")],
            leave=True,
            position=0,
        ):
            dataset, pid, seqid = fname.replace(".npz", "").split("_")
            pid = int(pid)
            seqid = int(seqid)
            full_fname = join(person_path, fname)
            data = np.load(full_fname)

            seq = PersonSequence(data, dataset=dataset, pid=pid)

            if dataset in self.dataset_lookup:
                self.dataset_lookup[dataset].append(seq)
            else:
                self.dataset_lookup[dataset] = [seq]

            key = (dataset, pid)
            if key in self.dataset_pid_lookup:
                self.dataset_pid_lookup[key].append(seq)
            else:
                self.dataset_pid_lookup[key] = [seq]

            for t in seq.frames:
                key = (dataset, t)
                if key in self.dataset_frame_lookup:
                    self.dataset_frame_lookup[key].append(seq)
                else:
                    self.dataset_frame_lookup[key] = [seq]

    def get_sequences(self, dataset: str) -> List[PersonSequence]:
        return self.dataset_lookup[dataset]

    def get_frame(self, dataset: str, frame: int, as_quaternions=False):
        key = (dataset, frame)
        if key in self.dataset_frame_lookup:
            return [
                seq.get_frame(frame, as_quaternions)
                for seq in self.dataset_frame_lookup[key]
            ]  # noqa E501
        else:
            return []

    def get_block3d(self, dataset: str, start_frame: int, end_frame: int):
        """
        :returns
            Poses3d: {n_frames x n_person x 29 x 3}
            Masks: {n_frames x n_person}
        """
        n_frames = end_frame - start_frame
        if n_frames < 1:
            raise ValueError(
                f"end frame {end_frame} must be larger than start frame {start_frame}"  # noqa E501
            )

        index2pid = {}
        pid2index = {}
        for i, pid in enumerate(pids_per_dataset[dataset]):
            index2pid[i] = pid
            pid2index[pid] = i

        n_person = len(pid2index)

        Mask = np.zeros((n_frames, n_person), dtype=np.float32)
        Poses3d = np.zeros((n_frames, n_person, 29, 3), dtype=np.float32)

        for i, frame in enumerate(range(start_frame, end_frame)):
            for entry in self.get_frame(dataset=dataset, frame=frame):
                pid = entry["pid"]
                j = pid2index[pid]
                pose3d = entry["pose3d"]
                Mask[i, j] = 1.0
                Poses3d[i, j] = pose3d

        return Mask, Poses3d

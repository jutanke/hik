from ndms.database import Data
from hik.data.person_sequence import PersonSequence
from hik.data.smpl import Body
from typing import List
from einops import repeat
from hik.transforms.transforms import normalize
from os.path import join, isfile
import numpy as np
from einops import rearrange


# you can define a transform function that is called for each motion word
# to for example normalize the data:
def transform(motion_word):
    """
    :param {kernel_size x dim*3}
    """
    motion_word = rearrange(motion_word, "t (j d) -> t j d", d=3)
    motion_word = normalize(
        motion_word,
        frame=0,
        zero_z=False,
        check_shape=False,
        allow_zero_z=True,  # noqa E501
    )  # noqa E501
    return rearrange(motion_word, "t j d -> t (j d)")


class NDMSData(Data):
    def __init__(
        self, seqs: List[PersonSequence], pid: int, body: Body, cache_dir=None
    ):
        # super().__init__()

        # find target seq
        target_seq = None
        for seq in seqs:
            if seq.pid == pid:
                target_seq = seq
                break

        if target_seq is None:
            raise ValueError(f"Cannot find target seq for pid {pid}")

        target_betas = target_seq.betas
        dataset = target_seq.dataset

        # self.selected_jids = list(range(29))
        self.selected_jids = [0, 1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21, 26, 27]

        self.data = []
        for seq in seqs:
            if seq.pid == pid:
                self.data.append(seq.poses3d[:, self.selected_jids])
            else:
                cached_fname = join(
                    cache_dir, f"ndms_ds{dataset}_pid{pid}_{seq.get_uid()}.npy"
                )  # noqa E501
                if isfile(cached_fname):
                    seq3d = np.load(cached_fname)
                    self.data.append(seq3d[:, self.selected_jids])
                else:
                    translation = seq.transforms[:, :3]
                    rotation = seq.transforms[:, 3:]
                    pose = seq.smpl
                    n_frames = len(pose)
                    sel_betas = repeat(target_betas, "d -> t d", t=n_frames)
                    seq3d = body.render_batch(
                        betas=sel_betas,
                        pose=pose,
                        translation=translation,
                        rotation=rotation,
                        use_tqdm=True,
                    )
                    np.save(cached_fname, seq3d)
                    self.data.append(seq3d[:, self.selected_jids])

    def n_dim(self):
        return len(self.selected_jids) * 3

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

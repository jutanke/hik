import unittest
import numpy as np
import numpy.linalg as la
from typing import List


def frames2segments(frames: List[int], return_indices=False, include_length=False):
    """
    :param frames: [1, 2, 3, 10, 11, 12, 13, ...]
    :return
        (1, 3), (11, 13), ..
    """
    segments = []
    indices = []
    if len(frames) > 0:
        frame = frames[0]
        i_start = 0
        for i in range(1, len(frames)):
            if frames[i - 1] + 1 < frames[i]:
                if include_length:
                    segments.append((frame, frames[i - 1], 1 + frames[i - 1] - frame))
                else:
                    segments.append((frame, frames[i - 1]))
                indices.append((i_start, i - 1))
                frame = frames[i]
                i_start = i
        # handle the last segment
        if include_length:
            segments.append((frame, frames[-1], 1 + frames[-1] - frame))
        else:
            segments.append((frame, frames[-1]))
        indices.append((i_start, len(frames) - 1))

    if return_indices:
        assert len(indices) == len(segments)
        return segments, indices

    return segments


def get_entry_from_batched_data(data, index: int):
    """ """
    entry = {}
    for key in data.keys():
        entry[key] = data[key][index]

    return entry


def get_splits(F: np.ndarray, length: int, stepsize=1):
    """
    :param F: [{frames}]
    :param non_overlapping: {bool}
        If True ensure that splits are not overlapping
    """
    split_starts = []
    for start, end in frames2segments(F):
        for t in range(start, end - length + 2, stepsize):
            split_starts.append(t)
    return split_starts


def unit(v):
    return v / la.norm(v)


# =======================================
# U N I T  T E S T S
# =======================================
class TestDataUtils(unittest.TestCase):
    def test_simple_splits_nonoverlapping(self):
        #    '--------'
        #                '--------'
        F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        splits = get_splits(F, length=4, stepsize=4)
        self.assertEqual(2, len(splits))
        self.assertListEqual([1, 5], splits)

    def test_simple_splits(self):
        #    '--------'
        #       '--------'
        #          '--------'
        #             '--------'
        #                '--------'
        #                   '--------'
        #                      '--------'
        F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        splits = get_splits(F, length=4, stepsize=1)
        self.assertEqual(7, len(splits))
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7], splits)

    def test_splits(self):
        F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
        splits = get_splits(F, length=4, stepsize=1)
        self.assertEqual(9, len(splits))
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 12, 13], splits)

        F = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 20, 21]
        splits = get_splits(F, length=4, stepsize=1)
        self.assertEqual(9, len(splits))
        self.assertListEqual([1, 2, 3, 4, 5, 6, 7, 12, 13], splits)

    def test_frames2segments_empty(self):
        """"""
        frames = []
        seg = frames2segments(frames)
        self.assertEqual(len(seg), 0)

    def test_frames2segments(self):
        """"""
        frames = [1, 2, 3, 4, 5, 6]
        seg = frames2segments(frames)
        self.assertEqual(len(seg), 1)
        self.assertSequenceEqual(seg[0], (1, 6))

        frames = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        seg = frames2segments(frames)
        self.assertEqual(len(seg), 2)
        self.assertSequenceEqual(seg[0], (1, 6))
        self.assertSequenceEqual(seg[1], (10, 13))

    def test_frames2segments_length(self):
        """"""
        frames = [1, 2, 3, 4, 5, 6]
        seg = frames2segments(frames, include_length=True)
        self.assertEqual(len(seg), 1)
        self.assertSequenceEqual(seg[0], (1, 6, 6))

        frames = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        seg = frames2segments(frames, include_length=True)
        self.assertEqual(len(seg), 2)
        self.assertSequenceEqual(seg[0], (1, 6, 6))
        self.assertSequenceEqual(seg[1], (10, 13, 4))

    def test_frames2segments_indices(self):
        """"""
        frames = [1, 2, 3, 4, 5, 6]
        seg, indices = frames2segments(frames, return_indices=True)
        self.assertEqual(len(seg), 1)
        self.assertSequenceEqual(seg[0], (1, 6))
        self.assertEqual(len(indices), 1)
        self.assertSequenceEqual(indices[0], (0, 5))

        frames = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13]
        seg, indices = frames2segments(frames, return_indices=True)
        self.assertEqual(len(seg), 2)
        self.assertSequenceEqual(seg[0], (1, 6))
        self.assertSequenceEqual(seg[1], (10, 13))
        self.assertEqual(len(indices), 2)
        self.assertSequenceEqual(indices[0], (0, 5))
        self.assertSequenceEqual(indices[1], (6, 9))


if __name__ == "__main__":
    unittest.main()

import numpy as np
from enum import IntEnum
import json
from hik.data.kitchen import Kitchen
from hik.data import PersonSequences
from os.path import join, isdir
from tqdm import tqdm
from hik.eval.mpjpe import calc_mpjpe
from typing import Dict


class EvaluationActionType(IntEnum):
    WALKING = 1
    SITTING_DOWN = 2
    WHITEBOARD = 3
    SINK = 4
    CUPBOARD = 5
    COFFEE = 6

    @staticmethod
    def enum2str(action):
        if action == EvaluationActionType.WALKING:
            return "walking"
        elif action == EvaluationActionType.SITTING_DOWN:
            return "sitting_down"
        elif action == EvaluationActionType.WHITEBOARD:
            return "whiteboard"
        elif action == EvaluationActionType.SINK:
            return "sink"
        elif action == EvaluationActionType.CUPBOARD:
            return "cupboard"
        elif action == EvaluationActionType.COFFEE:
            return "coffee"
        else:
            raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def str2enum(text):
        if text == "walking":
            return EvaluationActionType.WALKING
        elif text == "sitting_down":
            return EvaluationActionType.SITTING_DOWN
        elif text == "whiteboard":
            return EvaluationActionType.WHITEBOARD
        elif text == "sink":
            return EvaluationActionType.SINK
        elif text == "cupboard":
            return EvaluationActionType.CUPBOARD
        elif text == "coffee":
            return EvaluationActionType.COFFEE
        else:
            raise ValueError(f"Cannot find action to str'{text}'")

    @staticmethod
    def to_activity_names(action):
        if action == EvaluationActionType.WALKING:
            return "walking"
        elif action == EvaluationActionType.SITTING_DOWN:
            return "sitting down"
        elif action == EvaluationActionType.WHITEBOARD:
            return ["draw on whiteboard", "erase on whiteboard"]
        elif action == EvaluationActionType.SINK:
            return [
                "clean dish",
                "empty cup in sink",
                "put water in glass",
                "put water in kettle",
                "washing hands",
            ]
        elif action == EvaluationActionType.CUPBOARD:
            return [
                "open drawer",
                "open cupboard",
                "close cupboard",
                "close drawer",
            ]  # noqa E501
        elif action == EvaluationActionType.COFFEE:
            return [
                "make coffee",
                "place cup onto coffee machine",
                "press coffee button",
                "take cup from coffee machine",
            ]
        else:
            raise ValueError(f"Unknown action: {action}")


class Evaluator:
    def __init__(
        self, evaluation_json: str, dataset: str, data_path: str
    ) -> None:  # noqa E501
        if not isdir(data_path):
            raise ValueError(f"Cannot find data path '{data_path}'")

        with open(evaluation_json, "r") as f:
            self.data = json.load(f)

        if dataset not in self.data:
            raise ValueError(f"Cannot find dataset {dataset}...")

        self.dataset = dataset

        self.kitchen = Kitchen.load_for_dataset(
            dataset=dataset, data_location=join(data_path, "scenes")
        )

        self.person_seqs = PersonSequences(
            person_path=join(data_path, "poses")
        ).get_sequences(
            dataset=dataset
        )  # noqa E501

    def legacy_get_pids_per_sequence(self, action: str):
        """
        hacky way to get the pids lists for each sequence
        """
        all_pids = []

        def callback_fn(input):
            Poses3d_out = input["Poses3d_out"]
            pids = input["pids"]
            if action == input["action"]:
                all_pids.append(pids)
            return Poses3d_out

        self.execute3d(callback_fn=callback_fn, use_tqdm=False)
        return all_pids

    def execute3d(
        self, callback_fn, *, n_in=25 * 10, n_out=25 * 10, use_tqdm=True
    ):  # noqa E501
        """
        :param callback_fn(
            :param Seq: {n_frames x n_person x 29 x 3},
            :param Mask: {n_frames x n_person}
            :param pids: List[int]
            :param kitchen: Kitchen
            :param frame: int
        )
        """
        data = self.data[self.dataset]
        results = {}
        for action in data.keys():
            results[action] = []
            if use_tqdm:
                print(f"handle action: {action}")
            entries = data[action]
            for entry in tqdm(
                entries,
                position=0,
                leave=True,
                total=len(entries),
                disable=not use_tqdm,
            ):
                target_pid = entry["pid"]
                frame = entry["frame"]
                selected_seqs = []
                has_target = False
                for seq in self.person_seqs:
                    if seq.is_valid_at_frame(frame):
                        if seq.pid == target_pid:
                            has_target = True
                        selected_seqs.append(seq)
                if not has_target:
                    raise ValueError(
                        f"Cannot find target pid {target_pid} at frame {frame}"
                    )
                n_person = len(selected_seqs)
                n_frames = n_in + n_out
                Pids = []
                Mask = np.zeros((n_frames, n_person), dtype=np.float32)
                Poses3d = np.zeros(
                    (n_frames, n_person, 29, 3), dtype=np.float32
                )  # noqa E501

                frame_start = frame - n_in
                frame_end = frame + n_out

                for i, seq in enumerate(selected_seqs):
                    poses3d, mask = seq.get_range3d(
                        start_frame=frame_start, end_frame=frame_end
                    )
                    Pids.append(seq.pid)
                    Mask[:, i] = mask
                    Poses3d[:, i] = poses3d

                    if mask[n_in] < 0.5:
                        raise ValueError(
                            f"Cannot find pid{seq.pid} at frame {frame}"
                        )  # noqa E501

                Poses3d_in = np.ascontiguousarray(Poses3d[:n_in])
                Masks_in = np.ascontiguousarray(Mask[:n_in])
                Poses3d_out = np.ascontiguousarray(Poses3d[n_in:])
                Masks_out = np.ascontiguousarray(Mask[n_in:])

                Poses3d_out_pred = callback_fn(
                    {
                        "Poses3d_in": Poses3d_in,
                        "Masks_in": Masks_in,
                        "Poses3d_out": Poses3d_out,
                        "Masks_out": Masks_out,
                        "kitchen": self.kitchen,
                        "frames_in": list(range(frame - n_in, frame)),
                        "n_out": n_out,
                        "pids": Pids,
                        "action": action,
                    }
                )

                if len(Poses3d_out_pred.shape) != len(Poses3d_out.shape):
                    raise ValueError(
                        f"(1) Weird output shape: {Poses3d_out_pred.shape} but expected {Poses3d_out.shape}"  # noqa E501
                    )  # noqa E501
                for ii in range(len(Poses3d_out_pred.shape)):
                    if (
                        Poses3d_out_pred.shape[ii]
                        != Poses3d_out_pred.shape[ii]  # noqa E501
                    ):  # noqa E501
                        raise ValueError(
                            f"(2) Weird output shape: {Poses3d_out_pred.shape} but expected {Poses3d_out.shape}"  # noqa E501
                        )  # noqa E501

                results[action].append(
                    {
                        "Poses3d_in": Poses3d_in,
                        "Masks_in": Masks_in,
                        "Poses3d_out": Poses3d_out,
                        "Masks_out": Masks_out,
                        "frames_in": list(range(frame - n_in, frame)),
                        "Poses3d_out_pred": Poses3d_out_pred,
                        "target_pid": target_pid,
                        "pids": Pids,
                    }
                )

        return results

    def legacy_fix_pids_to_results(self, results: Dict):
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
        for action in results.keys():
            results_for_action = results[action]
            if "pids" not in results_for_action[0]:
                all_pids = self.legacy_get_pids_per_sequence(action=action)
                if len(results_for_action) != len(all_pids):
                    raise ValueError(
                        f"{len(all_pids)} vs {len(results_for_action)}"
                    )  # noqa E501
                for i, pids in enumerate(all_pids):
                    # sanity check
                    n_persons = results_for_action[i]["Poses3d_in"].shape[1]
                    if n_persons != len(pids):
                        raise ValueError(
                            f"Not fitting: {n_persons} vs {len(pids)}"
                        )  # noqa E501

                    target_pid = results_for_action[i]["target_pid"]
                    if target_pid not in pids:
                        raise ValueError(f"pid{target_pid} not in {pids}")

                    results_for_action[i]["pids"] = pids
        return results

    def calc_mpjpe(self, results: Dict):  # noqa F811
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
        return calc_mpjpe(results=results)

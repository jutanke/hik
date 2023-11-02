from typing import List

NUMBER_OF_SKELETON3D_JOINTS = 29

POINTS_PER_SQM = 50

LAST_FRAMES = {"A": 129722, "B": 177638, "C": 175556, "D": 177636}

pids_per_dataset = {
    "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19],
    "B": [
        7,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
    ],
    "C": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 40],
    "D": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        25,
        26,
    ],
}


DATASET_AND_PID_TO_GLOBAL_INDEX = {
    ("A", 1): 0,  # x
    ("C", 7): 0,  # x
    ("D", 3): 0,  # x
    ("B", 7): 0,  # x
    ("A", 2): 1,  # x
    ("A", 3): 2,  # x
    ("D", 20): 2,  # x
    ("B", 34): 2,  # x
    ("A", 4): 3,  # x
    ("B", 24): 3,  # x
    ("A", 5): 4,  # x
    ("C", 1): 4,  # x
    ("D", 1): 4,  # x
    ("B", 12): 4,  # x
    ("A", 6): 5,  # x
    ("C", 16): 5,  # x
    ("A", 7): 6,  # dont-know
    ("A", 8): 7,  # dont-know
    ("A", 9): 8,  # dont-know
    ("A", 10): 9,  # x
    ("C", 9): 9,  # x
    ("D", 7): 9,  # x
    ("B", 38): 9,  # x
    ("A", 11): 10,  # x
    ("C", 10): 10,  # x
    ("D", 22): 10,  # x
    ("B", 14): 10,  # x
    ("A", 12): 11,  # x
    ("A", 13): 12,  # x
    ("C", 4): 12,  # x
    ("D", 10): 12,  # x
    ("A", 14): 13,  # dont-know
    ("A", 15): 14,  # x
    ("A", 16): 15,  # dont-know
    ("A", 17): 16,  # x
    ("C", 14): 16,  # x
    ("A", 19): 17,  # x
    ("D", 23): 17,  # x
    ("C", 2): 18,  # x-Puppy
    ("D", 9): 18,  # x-Puppy
    ("C", 3): 19,  # x
    ("D", 8): 19,  # x
    ("B", 32): 19,  # x
    ("C", 5): 20,  # x
    ("D", 5): 20,  # x
    ("B", 15): 20,  # x
    ("C", 8): 21,  # x
    ("D", 21): 21,  # x
    ("B", 13): 21,  # x
    ("C", 11): 22,  # x
    ("D", 19): 22,  # x
    ("C", 12): 23,  # x
    ("D", 13): 23,  # x
    ("B", 25): 23,  # x
    ("C", 13): 24,  # no-idea
    ("C", 40): 24,  # no-idea
    ("C", 15): 25,  # x
    ("D", 18): 25,  # x
    ("D", 2): 26,  # dont-know
    ("D", 4): 27,  # dont-know
    ("D", 6): 28,  # dont-know
    ("D", 11): 29,  # x's puppy
    ("B", 29): 29,  # x's puppy
    ("D", 12): 30,  # x
    ("D", 15): 31,  # x
    ("D", 16): 32,  # dont-know
    ("D", 17): 33,  # dont-know
    ("D", 25): 34,  # x
    ("B", 33): 34,  # x
    ("D", 26): 35,  # dont-know
    ("B", 16): 36,  # x
    ("B", 17): 37,  # x
    ("B", 18): 38,  # dont-know
    ("B", 19): 39,  # dont-know
    ("B", 20): 40,  # x.
    ("B", 21): 41,  # dont-know
    ("B", 22): 42,  # x
    ("B", 23): 43,  # dont-know
    ("B", 26): 44,  # dont-know
    ("B", 27): 45,  # x
    ("B", 28): 46,  # dont-know
    ("B", 30): 47,  # dont-know
    ("B", 31): 48,  # dont-know
    ("B", 35): 49,  # dont-know
    ("B", 36): 50,  # dont-know
    ("B", 37): 51,  # dont-know
    ("B", 39): 52,  # dont-know
    ("B", 40): 53,  # dont-know
    ("B", 41): 54,  # dont-know
    ("B", 42): 55,  # dont-know
}


def pid_dataset_to_total_index(pid: int, dataset: str) -> int:
    """
    Total index maps a person and dataset to a unique id. We need this
    to be able to train data across various datasets at the same time.
    """
    return DATASET_AND_PID_TO_GLOBAL_INDEX[dataset, pid]


TOTAL_INDEX_COUNT = 36

activity2index = {
    "carry cake": 0,
    "carry cup": 1,
    "carry plate": 2,
    "carry whiteboard eraser": 3,
    "carry whiteboard marker": 4,
    "check water in coffee machine": 5,
    "clean countertop": 6,
    "clean dish": 7,
    "close cupboard": 8,
    "close dishwasher ": 9,
    "close drawer": 10,
    "close fridge": 11,
    "close window": 12,
    "cut cake in pieces": 13,
    "draw on whiteboard": 14,
    "drink": 15,
    "eat cake": 16,
    "eat fruit": 17,
    "empty cup in sink": 18,
    "empty dishwasher": 19,
    "empty ground from coffee machine": 20,
    "empty water from coffee machine": 21,
    "erase on whiteboard": 22,
    "fill coffee beens": 23,
    "fill water tank": 24,
    "fill water to coffee machine": 25,
    "fussball": 26,
    "kneeling": 27,
    "kneeling down": 28,
    "leaning": 29,
    "leaning down": 30,
    "listening": 31,
    "make coffee": 32,
    "mark coffee": 33,
    "open cupboard": 34,
    "open dishwasher": 35,
    "open drawer": 36,
    "open fridge": 37,
    "open window": 38,
    "peal fruit": 39,
    "phone call": 40,
    "place cake on plate": 41,
    "place cake on table": 42,
    "place cup onto coffee machine": 43,
    "place in dishwasher": 44,
    "place sheet onto whiteboard": 45,
    "place water tank in coffee machine": 46,
    "pour kettle": 47,
    "pour milk": 48,
    "press coffee button": 49,
    "put cake in fridge": 50,
    "put cup in microwave": 51,
    "put sugar in cup": 52,
    "put teabag in cup": 53,
    "put water in glass": 54,
    "put water in kettle": 55,
    "read paper": 56,
    "remove sheet from whiteboard": 57,
    "sitting": 58,
    "sitting down": 59,
    "squatting": 60,
    "standing": 61,
    "standing up": 62,
    "start dishwasher": 63,
    "start microwave": 64,
    "steps": 65,
    "take cake out of fridge": 66,
    "take cup from coffee machine": 67,
    "take cup out of microwave": 68,
    "take dish out of cupboard": 69,
    "take kettle": 70,
    "take milk": 71,
    "take piece of cake": 72,
    "take teabag": 73,
    "take water from sink": 74,
    "take water tank from coffee machine": 75,
    "talking": 76,
    "throw in trash": 77,
    "use laptop": 78,
    "use smartphone": 79,
    "walking": 80,
    "washing hands": 81,
}

index2activity = {}
for action, index in activity2index.items():
    index2activity[index] = action


def activity_to_name_list(act) -> List[str]:
    names = []
    for i, v in enumerate(act):
        if v > 0.5:
            names.append(index2activity[i])
    return names


KEEP_FACES_FOR_DATASET = {
    "A": {
        "collider0": [3, 5],
        "collider1": [1, 3],
        "collider2": [1, 3],
        "collider3": [4, 5],
        "collider5": [5],
    },
    "B": {
        "collider0": [4],
        "collider1": [1, 3],
        "collider99": [1],
    },
    "C": {
        "collider2": [3, 5],
        "collider3": [4],
        "collider0": [3, 5],
        "collider1": [3],
    },
    "D": {
        "collider2": [3],
        "collider3": [4, 5],
        "collider0": [1, 4],
        "collider4": [4, 5],
        "collider1": [1, 4],
    },
}

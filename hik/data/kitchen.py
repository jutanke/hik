import numpy as np
import numba as nb
import numpy.linalg as la
import math as m
import json
import os
from enum import IntEnum
from hik.data.constants import KEEP_FACES_FOR_DATASET, POINTS_PER_SQM, LAST_FRAMES
from os.path import join, isdir
from os import listdir
from multiprocessing.pool import ThreadPool
from typing import Dict, List

BOX_INDICES = np.array(
    [
        [0, 1, 2, 3],  # 0
        [1, 5, 6, 2],  # 1
        [5, 4, 7, 6],  # 2
        [1, 0, 4, 5],  # 3
        [2, 6, 7, 3],  # 4
        [3, 7, 4, 0],  # 5
    ],
    dtype=np.int64,
)


def face2sides(face):
    """
    :param face: [a, b, c, d]
    """
    return face[[0, 1]], face[[1, 2]], face[[2, 3]], face[[3, 0]]


@nb.njit(nb.types.UniTuple(nb.int64[:], 2)(nb.int64[:]), nogil=True)
def face2triangles(face):
    """
    :param face: [a, b, c, d]
    """
    a = np.array([0, 1, 2])
    b = np.array([2, 3, 0])
    return face[a], face[b]


def get_box_face_indices(face: int):
    """
            0---4
            | 3 |
    4...0---1---5---4...0
        | 0 | 1 | 2 |
    7...3---2---6---7...3
            | 4 |
            3---7
            | 5 |
            0---4
    """
    global BOX_INDICES
    assert 0 <= face < 6
    return BOX_INDICES[face]


class EnvironmentObject:
    def __init__(
        self,
        location,
        label,
        dataset="unknown",
        name="unknown",
        isbox=False,
        use_pointcloud=True,
        points_per_sqm=POINTS_PER_SQM,
        color="gray",
    ):
        self.dataset = dataset
        self.name = name
        self.location = location
        self.label = label
        self.isbox = isbox
        self.use_pointcloud = use_pointcloud
        self.points_per_sqm = points_per_sqm
        self.color = color

    def is_inside(self, pts3d):
        """
        :param pts3d: {n_points x 3}
        """
        if len(pts3d.shape) != 2 or pts3d.shape[1] != 3:
            raise ValueError(f"Weird shape: {pts3d.shape}")

        if self.isbox:
            box3d = self.location
            x_min = np.min(box3d[:, 0])
            y_min = np.min(box3d[:, 1])
            z_min = np.min(box3d[:, 2])
            x_max = np.max(box3d[:, 0])
            y_max = np.max(box3d[:, 1])
            z_max = np.max(box3d[:, 2])

            return (
                (x_min <= pts3d[:, 0])
                * (x_max >= pts3d[:, 0])
                * (y_min <= pts3d[:, 1])
                * (y_max >= pts3d[:, 1])
                * (z_min <= pts3d[:, 2])
                * (z_max >= pts3d[:, 2])
            )
        else:
            loc = np.expand_dims(self.location[:3], axis=0)
            r = self.location[3]
            d = la.norm(loc - pts3d, axis=1)
            return d <= r

    def query(self):
        D = self.location
        if self.use_pointcloud:
            if self.isbox:
                valid_faces = range(6)
                sides = []
                if (
                    self.dataset in KEEP_FACES_FOR_DATASET
                    and self.name in KEEP_FACES_FOR_DATASET[self.dataset]
                ):
                    valid_faces = []
                    for face in KEEP_FACES_FOR_DATASET[self.dataset][
                        self.name
                    ]:  # noqa E501
                        valid_faces.append(face)
                surfaces = []
                for fidx in valid_faces:
                    tri1, tri2 = face2triangles(BOX_INDICES[fidx])
                    surfaces.append(tri1)
                    surfaces.append(tri2)
                surfaces = np.array(surfaces)

                sides = []
                for fidx in valid_faces:
                    s1, s2, s3, s4 = face2sides(BOX_INDICES[fidx])
                    sides.append(tuple(s1.tolist()))
                    sides.append(tuple(s2.tolist()))
                    sides.append(tuple(s3.tolist()))
                    sides.append(tuple(s4.tolist()))
                sides = np.array(list(set(sides)), dtype=np.int64)
                return box_to_surface(
                    D,
                    points_per_sqm=self.points_per_sqm,
                    surfaces=surfaces,
                    sides=sides,
                )
            else:
                return cylinder_to_surface(D, points_per_sqm=self.points_per_sqm)
        else:
            return self.location


def cylinder_to_surface(D: np.ndarray, points_per_sqm: int):
    """"""
    mu = D[:3]
    r = D[3]
    Pts3d = [mu]
    circle_length = 2 * m.pi * r
    steps = max(6, int(m.ceil(m.sqrt(points_per_sqm) * circle_length)))

    theta = np.linspace(0, 3.141 * 2, num=steps)
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    z = np.zeros((steps,))
    circle = np.array([x, y, z]).transpose((1, 0))
    circle_top = circle + mu
    for pt in circle_top:
        Pts3d.append(pt)

    surface_area = m.pi * r**2
    n_points = max(5, int(m.ceil(points_per_sqm * surface_area)))

    random_rad = np.random.uniform(low=0.0000001, high=r, size=(n_points,))
    random_theta = np.random.uniform(low=0, high=2 * m.pi, size=(n_points,))

    x = random_rad * np.sin(random_theta)
    y = random_rad * np.cos(random_theta)
    z = np.zeros((n_points,))
    circle = np.array([x, y, z]).transpose((1, 0))
    circle_top = circle + mu
    for pt in circle_top:
        Pts3d.append(pt)
    return np.array(Pts3d, dtype=np.float32)


@nb.njit(
    nb.float32[:, :](nb.float32[:, :], nb.int64, nb.int64[:, :], nb.int64[:, :]),
    nogil=True,
)
def box_to_surface(
    D: np.ndarray, points_per_sqm: int, surfaces: np.ndarray, sides: np.ndarray
):
    """
    #   4--5    0---1  1---5  0---4  0---4  3---7  4---5
    #  /b /|    | A |  | B |  | C |  | D |  | E |  | F |
    # 0--1 6    2---3  2---6  1---5  3---7  2---6  7---6
    # |a |/
    # 3--2
    :param D: {8 x 3}
    """
    # |  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *
    # |  * Hey, this is a nice buffer-overflow to  *
    # v  * happen! Watch this thing with care!     *
    MAX_VAL = 10800
    # with 25pts/sqm we get the following TOTAL amount
    # of points:
    # A: 10328, B: 6400, C: 6646, D: 10783
    # So we can savely set the MAX_VAL to 10800

    Pts3d = np.empty((MAX_VAL, 3), dtype=np.float32)
    i = 0
    for a, b, c in surfaces:
        A = D[a]
        B = D[b]
        C = D[c]
        AB = D[a] - D[b]
        AC = D[a] - D[c]
        surface_area = la.norm(np.cross(AB, AC)) / 2
        n_points = max(5, int(m.ceil(points_per_sqm * surface_area)))
        for _ in range(n_points):
            wa = np.random.random()
            wb = np.random.random()
            wc = np.random.random()
            wabc = wa + wb + wc
            wa = wa / wabc
            wb = wb / wabc
            wc = wc / wabc
            Pts3d[i] = A * wa + B * wb + C * wc
            i += 1

    for a, b in sides:
        A = D[a]
        B = D[b]
        d = la.norm(A - B)
        steps = max(3, int(m.ceil(m.sqrt(points_per_sqm) * d)))
        ab = (B - A) / steps

        for ii in range(steps):
            Pts3d[i] = A + (ii + 1) * ab
            i += 1

    Pts3d = Pts3d[:i]

    return Pts3d


def load_object(data_location, name):
    data3d = np.load(join(data_location, f"{name}.npy"), mmap_mode="r").astype(
        "float32"
    )
    with open(join(data_location, f"{name}.json"), "r") as f:
        meta = json.load(f)
    return KitchenObject(meta, data3d)


def create_box(a, b, c, d):
    """
    :param a: {x, y}
    :param b: {x, y}
    """
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    A = [x1, y1, 0]
    B = [x2, y2, 0]
    C = [x3, y3, 0]
    D = [x4, y4, 0]
    E = [x1, y1, 0.1]
    F = [x2, y2, 0.1]
    G = [x3, y3, 0.1]
    H = [x4, y4, 0.1]
    return np.array([[A, B, C, D, E, F, G, H]], dtype=np.float32)


def get_out_of_bound_objects(dataset: str):
    """"""
    if dataset == "D":
        return [
            KitchenObject(
                meta={
                    "name": "oob1",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(-2, -12), b=(-12, -12), c=(-12, 20), d=(-2, 20)),
            ),
            KitchenObject(
                meta={
                    "name": "oob2",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(-4, -12), b=(-4, -3), c=(12, -3), d=(12, -12)),
            ),
            KitchenObject(
                meta={
                    "name": "oob3",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(4.53, -6), b=(4.53, 20), c=(12, 20), d=(12, -6)),
            ),
            KitchenObject(
                meta={
                    "name": "oob4",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(-4, 10.8), b=(-4, 20), c=(12, 20), d=(12, 10.8)),
            ),
        ]
    elif dataset == "C":
        return [
            KitchenObject(
                meta={
                    "name": "oob1",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(-10, -12), b=(-10, -2), c=(12, -2), d=(12, -12)),
            ),
            KitchenObject(
                meta={
                    "name": "oob2",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-10, -12), b=(-10, 20), c=(-1.0, 20), d=(-1.0, -12)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob3",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(4.55, -8), b=(4.55, 20), c=(16.0, 20), d=(16.0, -8)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob4",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(1.6, 5.0), b=(1.6, 20), c=(16.0, 20), d=(16.0, 5.0)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob5",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-2.0, 5.6),
                    b=(-2.0, 20.0),
                    c=(16.0, 20.0),
                    d=(16.0, 5.6),  # noqa E501
                ),
            ),
        ]
    elif dataset == "B":
        return [
            KitchenObject(
                meta={
                    "name": "oob1",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-12.0, -8),
                    b=(-12.0, -1.0),
                    c=(16.0, -1.0),
                    d=(16.0, -8.0),  # noqa E501
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob2",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-12.0, -8), b=(-12.0, 24), c=(-2.0, 24), d=(-2.0, -8.0)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob3",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-12.0, 9.9),
                    b=(-12.0, 24.0),
                    c=(20.0, 24.0),
                    d=(20.0, 9.9),  # noqa E501
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob4",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(5.6, 4.4), b=(3.2, 8), c=(6.0, 8), d=(6.4, 5.2)),
            ),
            KitchenObject(
                meta={
                    "name": "oob5",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(6.0, -8.0),
                    b=(6.0, 24.0),
                    c=(20.0, 24.0),
                    d=(20.0, -8.0),  # noqa E501
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob6",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(5.2, 3.6), b=(5.2, 4.8), c=(8.0, 4.8), d=(8.0, 3.6)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob7",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(3.2, 7.8), b=(3.2, 12.0), c=(8.0, 12.0), d=(8.0, 7.8)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob8",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-4.0, 4.5), b=(-4.0, 12.0), c=(0.0, 12.0), d=(0.0, 4.5)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob9",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-4.0, 2.8), b=(-4.0, 4.8), c=(0.0, 4.8), d=(-1.6, 2.8)
                ),
            ),
        ]
    elif dataset == "A":
        return [
            KitchenObject(
                meta={
                    "name": "oob1",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-12.0, -16.0),
                    b=(-12.0, -6.0),
                    c=(20.0, -6.0),
                    d=(20.0, -16.0),  # noqa E501
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob2",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(2, -16.0), b=(2, 20.0), c=(20.0, 20.0), d=(20.0, -16.0)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob3",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-17, -16.0), b=(-17, 20.0), c=(-7, 20.0), d=(-7, -16.0)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob4",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(
                    a=(-9, 0.1), b=(-9, 20.0), c=(-1.4, 20.0), d=(-1.4, 0.1)
                ),
            ),
            KitchenObject(
                meta={
                    "name": "oob5",
                    "shape": "box",
                    "type": "OUT_OF_BOUND",
                    "frame_multiplier": 0,
                },
                data3d=create_box(a=(-5, 6), b=(-5, 20.0), c=(15, 20.0), d=(15, 6)),
            ),
        ]
    else:
        raise NotImplementedError(f"dataset {dataset} missing..")


class KitchenObjectType(IntEnum):
    WHITEBOARD = 1
    MICROWAVE = 2
    KETTLE = 3
    COFFEE_MACHINE = 4
    TABLE = 5
    SITTABLE = 6
    CUPBOARD = 7
    OCCLUDER = 8
    DISHWASHER = 9
    DRAWER = 10
    SINK = 11
    TRASH = 12
    OUT_OF_BOUND = 13

    @staticmethod
    def name2type(name: str):
        """"""
        # fix some bugs that are from data generation
        name = name.replace("StaticObjectType.", "")
        if name == "WHITEBOARD":
            return KitchenObjectType.WHITEBOARD
        elif name == "MICROWAVE":
            return KitchenObjectType.MICROWAVE
        elif name == "KETTLE":
            return KitchenObjectType.KETTLE
        elif name == "COFFEE_MACHINE":
            return KitchenObjectType.COFFEE_MACHINE
        elif name == "TABLE":
            return KitchenObjectType.TABLE
        elif name == "SITTABLE":
            return KitchenObjectType.SITTABLE
        elif name == "CUPBOARD":
            return KitchenObjectType.CUPBOARD
        elif name == "OCCLUDER":
            return KitchenObjectType.OCCLUDER
        elif name == "DISHWASHER":
            return KitchenObjectType.DISHWASHER
        elif name == "DRAWER":
            return KitchenObjectType.DRAWER
        elif name == "SINK":
            return KitchenObjectType.SINK
        elif name == "TRASH":
            return KitchenObjectType.TRASH
        elif name == "OUT_OF_BOUND":
            return KitchenObjectType.OUT_OF_BOUND
        else:
            raise ValueError(f"Cannot assign the name  {name}")


class KitchenObject:
    def __init__(self, meta: Dict, data3d: np.ndarray):
        super().__init__()
        self.name = meta["name"]
        self.isbox = meta["shape"] == "box"
        self.obj_type = KitchenObjectType.name2type(meta["type"])
        self.obj_type_index = int(self.obj_type) - 1
        self.one_hot = np.zeros((13), dtype=np.float32)
        self.one_hot[self.obj_type_index] = 1.0
        self.data3d = data3d.astype("float32")
        self.alpha_weight = (
            0.0 if self.obj_type == KitchenObjectType.OUT_OF_BOUND else 1.0
        )

        # frame multiplier defines how to index a frame into the
        # data structure.
        # Originally, the data was annotated at 25/2Hz, so we have
        # to set the frame multiplier = 2 for 25Hz.
        # For static objects that do not move such as cupboards we
        # set the multiplier = 0 as the object is the same for all
        # frames.
        self.frame_multiplier = meta["frame_multiplier"]

        self.color = "gray"
        if self.obj_type == KitchenObjectType.SITTABLE:
            self.color = "red"
        if self.obj_type == KitchenObjectType.DISHWASHER:
            self.color = "green"
        elif self.obj_type == KitchenObjectType.TABLE:
            self.color = "cornflowerblue"
        elif self.obj_type == KitchenObjectType.TRASH:
            self.color = "black"
        elif self.obj_type == KitchenObjectType.COFFEE_MACHINE:
            self.color = "gold"
        elif self.obj_type == KitchenObjectType.WHITEBOARD:
            self.color = "orange"
        elif self.obj_type == KitchenObjectType.OCCLUDER:
            self.color = "lightgray"
        elif self.obj_type == KitchenObjectType.SINK:
            self.color = "darkblue"
        elif self.obj_type == KitchenObjectType.OUT_OF_BOUND:
            self.color = "pink"

    def get_data3d(self, frame: int):
        if self.frame_multiplier == 0:
            idx = 0
        else:
            idx = frame // self.frame_multiplier
        return self.data3d[idx]

    def get_one_hot_type(self):
        return self.one_hot

    def to_box(self):
        if not self.isbox:
            self.isbox = True
            new_data3d = []
            for i in range(len(self.data3d)):
                center = self.data3d[i, :3]
                radius = self.data3d[i, 3]

                # (-1,-1)-----( 1,-1)
                #    |           |
                #    |           |
                # (-1, 1)-----( 1, 1)
                top = []
                bottom = []
                bottom_mul = np.array([1, 1, 0])
                for pos in [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]:
                    pos = np.array(pos) * radius
                    top.append(center + pos)
                    bottom.append((center + pos) * bottom_mul)

                # convert 4 -> 8x3
                new_data3d.append(top + bottom)
            self.data3d = np.array(new_data3d)

    def plot(self, ax, frame: int, LW=1, color=None, alpha=0.8):
        """
        draw to {ax}
        """
        D = self.get_data3d(frame)
        color = self.color if color is None else color
        ALPHA = alpha
        ALPHA = ALPHA * self.alpha_weight

        if self.isbox:
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for a, b in connections:
                A = D[a]
                B = D[b]
                ax.plot(
                    [A[0], B[0]],
                    [A[1], B[1]],
                    [A[2], B[2]],
                    color=color,
                    linewidth=LW,
                    alpha=ALPHA,
                )
        else:
            center = np.expand_dims(D[:3], axis=0)
            r = D[3]
            num = 32
            theta = np.linspace(0, 3.141 * 2, num=num)
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            z = np.zeros((num,))
            circle = np.array([x, y, z]).transpose((1, 0))
            circle_top = circle + center
            circle_bottom = circle_top.copy()
            circle_bottom[:, 2] = 0
            connections = []
            for i in range(num - 1):
                connections.append((i, i + 1))
            connections.append((0, num - 1))
            for a, b in connections:
                A = circle_top[a]
                B = circle_top[b]
                ax.plot(
                    [A[0], B[0]],
                    [A[1], B[1]],
                    [A[2], B[2]],
                    linewidth=LW,
                    color=color,
                    alpha=ALPHA,
                )
            A = center[0]
            B = center[0].copy()
            B[2] = 0
            ax.plot(
                [A[0], B[0]],
                [A[1], B[1]],
                [A[2], B[2]],
                linewidth=LW,
                color=color,
                alpha=ALPHA,
            )


class Kitchen:
    """
    Contains the Scene
    """

    @staticmethod
    def load_for_dataset(dataset: str, data_location):
        if not isdir(data_location) and not data_location.startswith("/"):
            data_location = join(os.getcwd(), data_location)

        data_location = join(data_location, f"{dataset}_scene")
        assert isdir(data_location), data_location
        object_names = [
            (data_location, f[:-4])
            for f in listdir(data_location)
            if f.endswith(".npy")
        ]
        with ThreadPool(len(object_names)) as p:
            objects = p.starmap(load_object, object_names)

        if dataset == "A":
            xlim = [-8, 5]
            ylim = [-7, 6]
        elif dataset == "B":
            xlim = [-5, 8]
            ylim = [-2, 11]
        elif dataset == "C":
            xlim = [-5, 8]
            ylim = [-4, 9]
        elif dataset == "D":
            xlim = [-6, 8]
            ylim = [-3, 11]
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        objects += get_out_of_bound_objects(dataset=dataset)

        last_frame = LAST_FRAMES[dataset]
        return Kitchen(
            objects,
            xlim=xlim,
            ylim=ylim,
            last_frame=last_frame,
            dataset=dataset,  # noqa E501
        )

    def __init__(
        self,
        objects: List[KitchenObject],
        xlim,
        ylim,
        dataset: str,
        last_frame=-1,  # noqa E501
    ):
        super().__init__()
        self.dataset = dataset
        self.xlim = xlim
        self.ylim = ylim
        # self.zlim = [0, 13]
        self.zlim = [-5, 8]
        self.last_frame = last_frame
        self.objects = objects
        self.center3d = self._calculate_center3d(xlim=xlim, ylim=ylim)

    def _calculate_center3d(self, xlim, ylim):
        return np.array([[np.mean(xlim), np.mean(ylim), 0]], dtype=np.float32)

    def plot(self, ax, frame: int, LW=1, color=None, alpha=0.8):
        ax.set_zlim(self.zlim)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        for obj in self.objects:
            obj.plot(ax, frame, LW=LW, color=color, alpha=alpha)

    def get_environment(
        self, frame: int, ignore_oob=True, use_pointcloud=True
    ) -> List[EnvironmentObject]:
        if ignore_oob:
            env = []
            for obj in self.objects:
                if obj.obj_type != KitchenObjectType.OUT_OF_BOUND:
                    env.append(
                        EnvironmentObject(
                            name=obj.name,
                            dataset=self.dataset,
                            location=obj.get_data3d(frame),
                            label=obj.get_one_hot_type(),
                            isbox=obj.isbox,
                            use_pointcloud=use_pointcloud,
                            color=obj.color,
                        )
                    )
            return env
        else:
            return [
                EnvironmentObject(
                    name=obj.name,
                    dataset=self.dataset,
                    location=obj.get_data3d(frame),
                    label=obj.get_one_hot_type(),
                    isbox=obj.isbox,
                    use_pointcloud=use_pointcloud,
                    color=obj.color,
                )
                for obj in self.objects
            ]

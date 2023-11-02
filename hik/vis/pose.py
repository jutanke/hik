import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from tqdm import tqdm
from einops import rearrange
from hik.vis.utils import create_vis_path
from multiprocessing.pool import Pool


def plot_poses2d(
    ax,
    poses: np.ndarray,
    *,
    lcolor="cornflowerblue",
    rcolor="salmon",
    mcolor="gray",
    plot_jid=False,
    alpha=0.9,
    linewidth=3,
):
    for pose in poses:
        plot_poses2d(
            ax,
            pose,
            lcolor=lcolor,
            rcolor=rcolor,
            mcolor=mcolor,
            plot_jid=plot_jid,
            alpha=alpha,
            linewidth=linewidth,
        )


def plot_pose2d(
    ax,
    pose,
    *,
    lcolor="cornflowerblue",
    rcolor="salmon",
    mcolor="gray",
    plot_jid=False,
    alpha=0.9,
    linewidth=3,
):
    connect, left_jids, right_jids = get_meta(pose)

    for a, b in connect:
        color = mcolor
        if (a in left_jids and b not in right_jids) or (
            b in left_jids and a not in right_jids
        ):
            color = lcolor
        elif (a in right_jids and b not in left_jids) or (
            b in right_jids and a not in left_jids
        ):
            color = rcolor
        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    if plot_jid:
        for jid, pt2d in enumerate(pose):
            ax.text(pt2d[0], pt2d[1], str(jid))


def get_meta(pose):
    if pose.shape[0] == 29:
        connect = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            (16, 13),
            (17, 14),
            (13, 12),
            (14, 12),
            (12, 15),
            (15, 23),
            (24, 25),
            (24, 26),
            (25, 27),
            (26, 28),
        ]
        left_jids = set([2, 5, 8, 11, 14, 17, 19, 21, 25, 27])
        right_jids = set([1, 4, 7, 10, 13, 16, 18, 20, 26, 28])
    elif pose.shape[0] == 24:
        connect = [
            (1, 0),
            (0, 2),
            (1, 4),
            (4, 7),
            (7, 10),
            (2, 5),
            (5, 8),
            (8, 11),
            (16, 18),
            (18, 20),
            (17, 19),
            (19, 21),
            # (16, 13),
            (16, 12),
            # (17, 14),
            (17, 12),
            (13, 12),
            (14, 12),
            (12, 15),
            # (15, 23),
        ]
        left_jids = set([2, 5, 8, 11, 14, 17, 19, 21])
        right_jids = set([1, 4, 7, 10, 13, 16, 18, 20])
    elif pose.shape[0] == 17:
        connect = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
        ]
        left_jids = set([1, 3, 5, 6, 7, 11, 12, 13])
        right_jids = set([2, 4, 8, 9, 10, 14, 15, 16])
    elif pose.shape[0] == 13:
        # 0 1 2 3 4
        # 5 6 [x x] 7 8
        # 9 10 [x x] 11 12
        connect = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
        ]  # noqa E501
        left_jids = set([1, 3, 5, 6, 9, 10])
        right_jids = set([2, 4, 7, 8, 11, 12])
    else:
        raise ValueError(f"Shape is weird: {pose.shape}")

    return connect, left_jids, right_jids


def plot_poses(
    ax,
    poses,
    *,
    lcolor="cornflowerblue",
    rcolor="salmon",
    mcolor="gray",
    plot_jid=False,
    dont_plot_jids=None,
    alpha=0.9,
    linewidth=3,
):
    for pose in poses:
        plot_pose(
            ax,
            pose,
            lcolor=lcolor,
            rcolor=rcolor,
            mcolor=mcolor,
            plot_jid=plot_jid,
            dont_plot_jids=dont_plot_jids,
            alpha=alpha,
            linewidth=linewidth,
        )


def plot_pose(
    ax,
    pose,
    *,
    lcolor="cornflowerblue",
    rcolor="salmon",
    mcolor="gray",
    plot_jid=False,
    dont_plot_jids=None,
    alpha=0.9,
    linewidth=3,
):
    """
    :param pose: {29x3} OR {17x3}
    """
    if dont_plot_jids is None:
        dont_plot_jids = []
    dont_plot_jids = set(dont_plot_jids)

    connect, left_jids, right_jids = get_meta(pose)

    for a, b in connect:
        color = mcolor
        if (a in left_jids and b not in right_jids) or (
            b in left_jids and a not in right_jids
        ):
            color = lcolor
        elif (a in right_jids and b not in left_jids) or (
            b in right_jids and a not in left_jids
        ):
            color = rcolor
        ax.plot(
            [pose[a, 0], pose[b, 0]],
            [pose[a, 1], pose[b, 1]],
            [pose[a, 2], pose[b, 2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    if plot_jid:
        for jid, pt3d in enumerate(pose):
            if jid not in dont_plot_jids:
                ax.text(pt3d[0], pt3d[1], pt3d[2], str(jid))


def set_lims(ax, seq):
    """
    :param seq: {n x j x d}
    """
    xlim, ylim, zlim = get_lims(seq)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


def get_lims(seq):
    """
    :param seq: {n x j x d}
    """
    if len(seq.shape) == 2:
        seq = rearrange(seq, "t (j d) -> t j d", d=3)

    x_min = np.min(seq[:, :, 0])
    x_max = np.max(seq[:, :, 0])
    y_min = np.min(seq[:, :, 1])
    y_max = np.max(seq[:, :, 1])

    ll = max(max(x_max - x_min, y_max - y_min), 2.0)

    x_min = (x_max + x_min) / 2 - ll / 2
    x_max = (x_max + x_min) / 2 + ll / 2
    y_min = (y_max + y_min) / 2 - ll / 2
    y_max = (y_max + y_min) / 2 + ll / 2

    return ([x_min, x_max], [y_min, y_max], [0, ll])


def _vis_single_frame(data):
    """
    :param data: {
        "fname": str,
        "colors": (lcolor, rcolor),
        "pose": pose {29 x 3}
        "alpha": float,
        "xlim" [...],
        "ylim" [...],
        "zlim" [...],
        "t": int
    }
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"frame {data['t']}")
    ax.set_xlim(data["xlim"])
    ax.set_ylim(data["ylim"])
    ax.set_zlim(data["zlim"])
    if data["show_origin"]:
        ax.plot([-1, 1], [0, 0], [0, 0], color="gray", alpha=0.4)
        ax.plot([0, 0], [-1, 1], [0, 0], color="gray", alpha=0.4)
        ax.plot([0, 0], [0, 0], [-1, 1], color="gray", alpha=0.4)
    lcolor, rcolor = data["colors"]
    plot_pose(
        ax,
        pose=data["pose"],
        alpha=data["alpha"],
        lcolor=lcolor,
        rcolor=rcolor,
    )
    plt.savefig(data["fname"])
    return 0


def vis_seq(
    path: str,
    seq: np.ndarray,
    *,
    alpha=0.9,
    show_origin=True,
    forecasting_starts_at=-1,
    colors=("cornflowerblue", "salmon"),
    fc_colors=("orange", "green"),
    allow_multithreaded=True,
):
    """
    :param seq: {n_frames x 29 x 3}
    """
    Seq = None
    if isinstance(seq, list):
        Seq = seq
        seq = Seq[0]
    Colors = None
    FC_Colors = None
    if isinstance(colors, list):
        Colors = colors
    else:
        Colors = [colors]
    if isinstance(fc_colors, list):
        FC_Colors = fc_colors
    else:
        FC_Colors = [fc_colors]

    if forecasting_starts_at == -1:
        forecasting_starts_at = len(seq) + 1

    print(f"visualize {seq.shape} to {path}")
    create_vis_path(path)

    if len(seq.shape) == 2:
        seq = rearrange(seq, "t (j d) -> t j d", d=3)

    x_min = np.min(seq[:, :, 0])
    x_max = np.max(seq[:, :, 0])
    y_min = np.min(seq[:, :, 1])
    y_max = np.max(seq[:, :, 1])

    ll = max(max(x_max - x_min, y_max - y_min), 2.0)

    x_min = (x_max + x_min) / 2 - ll / 2
    x_max = (x_max + x_min) / 2 + ll / 2
    y_min = (y_max + y_min) / 2 - ll / 2
    y_max = (y_max + y_min) / 2 + ll / 2
    # -- b

    if allow_multithreaded and Seq is None:
        Data = []
        for t, pose in tqdm(enumerate(seq)):
            if t >= forecasting_starts_at:
                lcolor, rcolor = fc_colors
            else:
                lcolor, rcolor = colors
            Data.append(
                {
                    "fname": join(path, "frame%05d.png" % t),
                    "colors": (lcolor, rcolor),
                    "alpha": alpha,
                    "pose": pose,
                    "xlim": [x_min, x_max],
                    "ylim": [y_min, y_max],
                    "zlim": [0, ll],
                    "t": t,
                    "show_origin": show_origin,
                }
            )
        with Pool(10) as p:
            _ = list(tqdm(p.imap(_vis_single_frame, Data), total=len(Data)))

    else:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        for t, pose in tqdm(enumerate(seq)):
            ax.clear()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"frame {t}")
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            if show_origin:
                ax.plot([-1, 1], [0, 0], [0, 0], color="gray", alpha=0.4)
                ax.plot([0, 0], [-1, 1], [0, 0], color="gray", alpha=0.4)
                ax.plot([0, 0], [0, 0], [-1, 1], color="gray", alpha=0.4)
            ax.set_zlim([0, ll])
            if Seq is None:
                if t >= forecasting_starts_at:
                    lcolor, rcolor = fc_colors
                else:
                    lcolor, rcolor = colors
                plot_pose(
                    ax, pose=pose, alpha=alpha, lcolor=lcolor, rcolor=rcolor
                )  # noqa E501
            else:
                for ii in range(len(Seq)):
                    if t >= forecasting_starts_at:
                        lcolor, rcolor = FC_Colors[ii % len(FC_Colors)]
                    else:
                        lcolor, rcolor = Colors[ii % len(Colors)]
                    pose = Seq[ii][t]
                    if len(pose.shape) == 1:
                        pose = rearrange(pose, "(j d) -> j d", d=3)
                    plot_pose(
                        ax,
                        pose=pose,
                        alpha=alpha,
                        lcolor=lcolor,
                        rcolor=rcolor,  # noqa E501
                    )  # noqa E501

            fname = join(path, "frame%05d.png" % t)
            plt.savefig(fname)

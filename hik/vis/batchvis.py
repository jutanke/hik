from hik.vis.pose import plot_pose
from hik.data.kitchen import Kitchen
import numpy as np
from hik.vis.utils import save_delete_vid_dir, to_mp4
from hik.data.constants import pids_per_dataset
from tqdm import tqdm
from os.path import join
from os import makedirs
import matplotlib.pylab as plt
from multiprocessing.pool import Pool


def render_to_file(entry):
    """
    :param output: {
        "frame": frame,
        "fname": fname,
        "objs3d": objs3d,
        "poses": poses,
        "draw_pids",
        "view_init",
        "pid2color",
        "pid2alpha",
        "lims"
    }
    """
    frame = entry["frame"]
    fname = entry["fname"]
    objs3d = entry["objs3d"]
    poses = entry["poses"]
    xlim, ylim, zlim = entry["lims"]
    view_init = entry["view_init"]
    pid2color = entry["pid2color"]
    if pid2color is None:
        pid2color = {}
    pid2alpha = entry["pid2alpha"]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(*view_init)

    dim = 2
    ax.set_xlim([xlim[0] + dim, xlim[1] - dim])
    ax.set_ylim([ylim[0] + dim, ylim[1] - dim])
    ax.set_zlim([zlim[0] + dim, zlim[1] - dim])

    for pts3d, color in objs3d:
        ax.scatter(
            pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], color=color, alpha=0.1, s=1
        )  # noq E501

    for pose in poses:
        pid = pose["pid"]
        if pid in pid2color:
            lcolor, rcolor = pid2color[pid]
        else:
            lcolor = "cornflowerblue"
            rcolor = "salmon"
        if pid2alpha is None:
            alpha = 0.9
        else:
            if pid in pid2alpha:
                alpha = pid2alpha[pid]
            else:
                alpha = 0.2

        plot_pose(ax, pose["pose"], lcolor=lcolor, rcolor=rcolor, alpha=alpha)
        if entry["draw_pids"]:
            x, y, z = pose["pose"][0]
            ax.text(x, y, z, str(pid))

    ax.set_title(f"\n\nFrame {frame}")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return 0


def vis_batch(
    path: str,
    kitchen: Kitchen,
    data,
    *,
    draw_pids=True,
    rotate=True,
    forecast_frame: int = -1,
    elev=45,
):
    """
    :param data: {
        "poses3d": {n_frames x n_person x 29 x 3}
        "masks": {n_frames x n_person}
        "start_frames": {int}
        "pids": List[int]  # this is required if n_person is NOT all persons
                             of the dataset
        "pid2color": Dict
        "pid2color_future": Dict
        "pid2alpha": Dict
    }
    """
    index2pid = {}
    pid2index = {}
    for i, pid in enumerate(pids_per_dataset[kitchen.dataset]):
        index2pid[i] = pid
        pid2index[pid] = i

    save_delete_vid_dir(path=path)
    makedirs(path)

    poses3d = data["poses3d"]
    masks = data["masks"]

    pid2color = data["pid2color"] if "pid2color" in data else None
    pid2alpha = data["pid2alpha"] if "pid2alpha" in data else None
    pid2color_future = (
        data["pid2color_future"] if "pid2color_future" in data else None
    )  # noqa E501
    if pid2color_future is None:
        pid2color_future = {}
        for pid in pids_per_dataset[kitchen.dataset]:
            pid2color_future[pid] = ("orange", "green")

    if len(poses3d.shape) != 4 or not (
        poses3d.shape[2] == 29 and poses3d.shape[3] == 3
    ):
        raise ValueError(
            f"weird 3d poses shape: {poses3d.shape} but expect [t x p x 29 x 3]"  # noqa E501
        )  # noqa E501

    # verify that we have exactly the right amount of persons!
    n_person = poses3d.shape[1]
    n_person_in_dataset = len(index2pid)
    n_frames = len(poses3d)
    if forecast_frame == -1:
        forecast_frame = n_frames + 1

    if n_person < n_person_in_dataset:
        n_frames = poses3d.shape[0]

        if "pids" not in data:
            raise ValueError(
                "We require a 'pids' entry as the number of persons in the sequence is smaller than the total one in the dataset"  # noqa E501
            )
        pids = data["pids"]

        poses3d_ = np.zeros(
            (n_frames, n_person_in_dataset, 29, 3), dtype=np.float32
        )  # noqa E501
        masks_ = np.zeros((n_frames, n_person_in_dataset), dtype=np.float32)

        for i, pid in enumerate(pids):
            j = pid2index[pid]
            poses3d_[:, j] = poses3d[:, i]
            masks_[:, j] = masks[:, i]

        poses3d = poses3d_
        masks = masks_

    elif n_person > n_person_in_dataset:
        raise ValueError(
            f"poses {poses3d.shape} does not fit with total number of persons #{len(index2pid)}"  # noqa E501
        )

    start_frame = data["start_frames"]

    data = []

    if rotate:
        rotation = np.linspace(0, 360, num=n_frames)
    else:
        rotation = [45] * n_frames

    for i in range(n_frames):
        fname = join(path, "frame%06d.png" % i)
        frame = start_frame + i
        objs3d = [
            (obj.query(), obj.color)
            for obj in kitchen.get_environment(frame=frame)  # noqa E501
        ]
        poses = []
        for j, (mask, pose) in enumerate(zip(masks[i], poses3d[i])):
            if mask > 0.5:
                pid = index2pid[j]
                poses.append({"pid": pid, "pose": pose})

        data.append(
            {
                "frame": frame,
                "fname": fname,
                "objs3d": objs3d,
                "pid2color": pid2color
                if i < forecast_frame
                else pid2color_future,  # noqa E501
                "poses": poses,
                "draw_pids": draw_pids,
                "pid2alpha": pid2alpha,
                "view_init": [elev, rotation[i]],
                "lims": [kitchen.xlim, kitchen.ylim, kitchen.zlim],
            }
        )

    with Pool(9) as p:
        _ = list(tqdm(p.imap(render_to_file, data), total=len(data)))

    to_mp4(path=path)

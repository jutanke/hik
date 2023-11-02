from os.path import isdir, join
from os import listdir, makedirs
import shutil
import cv2


def save_delete_vid_dir(path: str):
    if isdir(path):
        allowed_file_types = [".mp4", ".png"]
        for f in listdir(path):
            is_fine_to_be_deleted = False
            for end in allowed_file_types:
                if f.endswith(end):
                    is_fine_to_be_deleted = True
                    break
            if not is_fine_to_be_deleted:
                raise ValueError(f"Cannot delete {path} as there is a weird file: {f}")  # noqa E501
        shutil.rmtree(path)


def create_vis_path(path: str):
    save_delete_vid_dir(path)
    makedirs(path)


def to_mp4(path: str, fps=25):
    if path[-1] == "/":
        path = path[:-1]
    assert isdir(path), path
    img_array = []
    for fname in sorted(
        [join(path, fname)
         for fname in listdir(path) if fname.endswith(".png")]
    ):
        img = cv2.imread(fname)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    root = "/".join(path.split("/")[:-1])
    name = path.split("/")[-1] + ".mp4"
    vid_fname = join(root, name)
    out = cv2.VideoWriter(
        vid_fname, cv2.VideoWriter_fourcc(*"MP4V"), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    save_delete_vid_dir(path)

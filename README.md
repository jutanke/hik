# [NeurIPS 2023 Dataset and Benchmark Track] Humans in Kitchens: A Dataset for Multi-Person Human Motion Forecasting with Scene Context
![](https://github.com/jutanke/hik/blob/main/documentation/data/A_32450.mp4.gif)

Official API for our [dataset](https://github.com/jutanke/hik/blob/main/documentation/data/neurips_camready.pdf) submission to NeurIPS 2023.

## Abstract
Forecasting human motion of multiple persons is very challenging. It requires to model the interactions between humans and the interactions with objects and the environment. For example, a person might want to make a coffee, but if the coffee machine is already occupied the person will have to wait. These complex relations between scene geometry and persons arise constantly in our daily lives, and models that wish to accurately forecast human behavior will have to take them into consideration. To facilitate research in this direction, we propose Humans in Kitchens, a large-scale multi-person human motion dataset with annotated 3D human poses, scene geometry and activities per person and frame. Our dataset consists of over 7.3h recorded data of up to 16 persons at the same time in four kitchen scenes, with more than 4M annotated human poses, represented by a parametric 3D body model. In addition, dynamic scene geometry and objects like chair or cupboard are annotated per frame. As first benchmarks, we propose two protocols for short-term and long-term human motion forecasting.


## Dataset
Download the [dataset](https://drive.google.com/file/d/1ctMoU8PcBSUQtJWuLIPhomhEsh3Pg0bf/view?usp=drive_link) (**preliminary version 0.0.1**).
The dataset has the following structure
* `data/`
    * `body_models/`
        * `SMPLX_NEUTRAL.npz`
    * `poses/`
        * `{dataset}_{pid}_{seqid}.npz`
    * `scenes/`
        * `{dataset}_scene/`
            * `{object_name}.json`
            * `{object_name}.npy`

In the rest of the documentation we assume that the folder `data/` is located at `/path/to/dataset/data/`.

## Documentation
### Install
Install the tool via pip:
```
pip install git+https://github.com/jutanke/hik.git
```

For evaluation, [ndms](https://github.com/jutanke/ndms) has to be installed too:
```
pip install git+https://github.com/jutanke/ndms.git
```

### Usage
[API](https://github.com/jutanke/hik/blob/main/documentation/api.md)
```python
import matplotlib.pylab as plt
import numpy as np

from hik.data.kitchen import Kitchen
from hik.data import PersonSequences
from hik.vis import plot_pose

# load geometry
kitchen = Kitchen.load_for_dataset(
    dataset=dataset,
    data_location="/path/to/dataset/data/scenes"
)

# load poses
person_seqs = PersonSequences(
    person_path="/path/to/dataset/data/poses"
)

smplx_path = "/path/to/dataset/data/body_models"

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

dataset = "C"  # ["A", "B", "C", "D"]
frame = 12345

kitchen.plot(ax, frame)
for person in person_seqs.get_frame(dataset, frame):
    plot_pose(ax, person["pose3d"], linewidth=1)

ax.axis('off')
```

### 3D Skeleton

| 3D joints (body)  | 3D joints (head) |
| ------------- | ------------- |
| ![](https://github.com/jutanke/hik/blob/main/documentation/data/skeleton_jids.png)  | ![](https://github.com/jutanke/hik/blob/main/documentation/data/skeleton_jids_head.png) |


## Statistics

![](https://github.com/jutanke/hik/blob/main/documentation/data/count_pose.png)

### Heatmaps
![](https://github.com/jutanke/hik/blob/main/documentation/data/heatmaps.png)
![](https://github.com/jutanke/hik/blob/main/documentation/data/heatmaps2.png)
![](https://github.com/jutanke/hik/blob/main/documentation/data/heatmaps3.png)

## Sample Sequences

![](https://github.com/jutanke/hik/blob/main/documentation/data/A_32450.mp4.gif)
![](https://github.com/jutanke/hik/blob/main/documentation/data/outB.mp4.gif)
![](https://github.com/jutanke/hik/blob/main/documentation/data/outC.mp4.gif)
![](https://github.com/jutanke/hik/blob/main/documentation/data/outD.mp4.gif)

## Entire Dataset
Speed-up sequences for all 4 kitchens

### Dataset A
![](https://github.com/jutanke/hik/blob/main/documentation/data/A_fast2.mp4.gif)
### Dataset B
![](https://github.com/jutanke/hik/blob/main/documentation/data/B_fast.mp4.gif)
### Dataset C
![](https://github.com/jutanke/hik/blob/main/documentation/data/C_fast.mp4.gif)
### Dataset D
![](https://github.com/jutanke/hik/blob/main/documentation/data/D_fast.mp4.gif)

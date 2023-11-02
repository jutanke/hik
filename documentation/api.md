# API

## Poses

```python
from hik.data import PersonSequences
from hik.data.constants import activity2index


smplx_path = "/path/to/dataset/data/body_models"


# The dataset is made up of 4 kitchens: A, B, C, D.
# The 'PersonSequences' class loads all poses for 
# all people in all kitchens.
person_seqs = PersonSequences(
    person_path="/path/to/dataset/data/poses"
)

# all sequences of a specific kitchen can be queried
# as follows
# seqs <= List[hik.data.person_sequences.PersonSequence]
seqs = person_seqs.get_sequences("A")

seq = seqs[0]  # hik.data.person_sequences.PersonSequence

# - - - - - - - - - -
# META
# - - - - - - - - - -
n_frames = len(seq)  # how many frames? @25Hz

# the kitchen-global per-frame frame number . These frames
# are guaranteed to be adjacent, i.e. frames will always
# increment by 1
seq.frames  # <- n_frames

# - - - - - - - - - -
# Activities
# - - - - - - - - - -
# 82 Activities as binary encoding.
# If the activity is active at that
# frame its 1, otherwise 0
seq.act  # <- n_frames x 82

# for example, check if the person is doing a certain
# activity as a one-hot encoding
is_walking = seq.act[:, activity2index["walking"]]  # <- n_frames

is_open_cupboard = seq.act[:, activity2index["open cupboard"]]

is_use_smartphone = seq.act[:, activity2index["use smartphone"]]

# - - - - - - - - - -
# SMPL
# - - - - - - - - - -
# SMPL pose parameters for SMPL. The rotations are
# represented as axis-angles
seq.smpl  # <- n_frames x 21 x 3

# same as 'smpl' but the axis-angles are represented
# as quaternions
smpl_quat = seq.get_smpl_as_quaterions()

# The global transformation of SMPL is encoded in
# transforms where the first 3 values represent the
# translation and the last 3 values represent the
# rotation as axis-angle
seq.transforms  # <- n_frames x 6

# same as 'transforms' but the rotational vector is
# represented as quaternion
transforms_quat = seq.get_transforms_as_quaternion()

# SMPL shape parameters
seq.betas  # <- 10,

# - - - - - - - - - -
# Poses-3D
# - - - - - - - - - -
# As a convenient function we extract the 3D poses
# from SMPL and provide them directly.
seq.poses3d  # <- n_frames x 29 x 3 

# - - - - - - - - - -
# API
# - - - - - - - - - -

# check if the given frame is within this sequence
seq.is_valid_at_frame(frame=9999)

# get all frames for this sequence where the person
# is walking and sitting
seq.get_frames_where_action(actions=["walking", "sitting"])

```

## Kitchens

```python
from hik.data.kitchen import Kitchen

# load geometry
kitchen = Kitchen.load_for_dataset(
    dataset="A",
    data_location="/path/to/dataset/data/scenes"
)

# get the kitchen geometry for the given frame as
# a list of 'hik.data.kitchen.EnvironmentObject'
env_objects = kitchen.get_environment(frame=999)

env_object = env_objects[0]  # hik.data.kitchen.EnvironmentObject
# each object has a label attached to it which determines 
# the class:
# WHITEBOARD = 1
# MICROWAVE = 2
# KETTLE = 3
# COFFEE_MACHINE = 4
# TABLE = 5
# SITTABLE = 6
# CUPBOARD = 7
# OCCLUDER = 8
# DISHWASHER = 9
# DRAWER = 10
# SINK = 11
# TRASH = 12
# OUT_OF_BOUND = 13
env_object.label.shape  # <- 13,

# the location is either 8 points of a box (8 x 3)
# for box-shaped objects or a 3d point + radius (4) 
# for cylinder objects.
# Note that cylinder objects always stand on the ground
# (z=0)
env_object.location

# create 1000 random points
pts3d = np.random.uniform(size=(1000, 3))
# check if any of the points are inside
# the object...
env_object.is_inside(pts3d)

```
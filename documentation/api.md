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
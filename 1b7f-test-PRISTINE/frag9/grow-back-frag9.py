"""
Grow back dinuc fragment 9 by superposition on the poses of fragment 10
"""

import numpy as np
from crocodile.nuc.all_fit import all_fit
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries

refe_ppdb = np.load("1b7f-rna-aa.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

ovRMSD = 0.643 + 0.01
cRMSD = 0.381 + 0.01

seq = refe.get_sequence(9, fraglen=2)
prev_seq = refe.get_sequence(10, fraglen=2)
print(seq, prev_seq)
libf = dinucleotide_libraries[seq]
prev_libf = dinucleotide_libraries[prev_seq]

prev_poses = np.load("poses-prev.npy")
libf.load_rotaconformers()

print(len(prev_poses))
lib = libf.create(
    pdb_code="1b7f", nucleotide_mask=[False, True], with_rotaconformers=True
)
prev_lib = prev_libf.create(pdb_code="1b7f", nucleotide_mask=[True, False])
new_poses = []
for prev_pose in prev_poses:
    prev_pose_conf_coors = prev_lib.coordinates[prev_pose["conformer"]]
    prev_pose_coors = (
        prev_pose_conf_coors.dot(prev_pose["rotation"]) + prev_pose["offset"]
    )
    curr_new_poses = all_fit(
        prev_pose_coors,
        fragment_library=lib,
        rmsd_threshold=ovRMSD,
        conformer_rmsd_threshold=cRMSD,
        rotamer_precision=0.5,
        grid_spacing=np.sqrt(3) / 3,
        return_rotamer_indices=True,
    )
    print(len(new_poses), len(curr_new_poses))
    new_poses.append(curr_new_poses)

nposes = sum([len(new_p) for new_p in new_poses])
poses = np.empty(nposes, dtype=prev_poses.dtype)
pos = 0
with open("poses0-conf-origins.txt", "w") as f:
    for ori, new_p in enumerate(new_poses):
        if not len(new_p):
            continue
        for n in range(len(new_p)):
            print(ori + 1, file=f)
        poses[pos : pos + len(new_p)] = new_p
        pos += len(new_p)

lib2 = libf.create(pdb_code="1b7f")
refe_coors = refe.get_coordinates(9, fraglen=2)
poses_coors_conf = lib2.coordinates[poses["conformer"]]
poses_coors = (
    np.einsum("ikj,ijl->ikl", poses_coors_conf, poses["rotation"])
    + poses["offset"][:, None, :]
)
dif = poses_coors - refe_coors
rmsd = np.sqrt(np.einsum("ijk,ijk->i", dif, dif) / len(refe_coors))
print(len(poses), rmsd.min())

np.savetxt("poses.rmsd", rmsd, fmt="%.3f")
np.save("poses0.npy", poses)

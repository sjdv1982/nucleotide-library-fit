"""
Grow back dinuc fragment 7 by superposition on the poses of fragment 6

Takes ~5 mins to build the 10 million poses.
Small fraction (1.05 in 10 000) is missed.
"""

import numpy as np
from crocodile.nuc.all_fit import all_fit
from crocodile.nuc.reference import Reference
from library_config import mononucleotide_templates, dinucleotide_libraries
from tqdm import tqdm

refe_ppdb = np.load("1b7f-rna-aa.npy")
refe = Reference(
    ppdb=refe_ppdb,
    mononucleotide_templates=mononucleotide_templates,
    rna=True,
    ignore_unknown=False,
    ignore_missing=False,
    ignore_reordered=False,
)

ovRMSD = 1.188 + 0.01
cRMSD = 0.911 + 0.01

seq = refe.get_sequence(7, fraglen=2)
prev_seq = refe.get_sequence(6, fraglen=2)
print(seq, prev_seq)
libf = dinucleotide_libraries[seq]
prev_libf = dinucleotide_libraries[prev_seq]

prev_poses = np.load("poses-prev.npy")
libf.load_rotaconformers()

print(len(prev_poses))
lib = libf.create(
    pdb_code="1b7f", nucleotide_mask=[True, False], with_rotaconformers=True
)
prev_lib = prev_libf.create(pdb_code="1b7f", nucleotide_mask=[False, True])
new_poses = []
for prev_pose in tqdm(prev_poses):
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

libf.unload_rotaconformers()
del lib

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

np.save("poses0.npy", poses)
del new_poses

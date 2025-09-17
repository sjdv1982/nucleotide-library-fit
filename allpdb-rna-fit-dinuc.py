from seamless import Buffer
from tqdm import tqdm

import numpy as np

import sys

motif = sys.argv[1]
assert (
    len(motif) == 2
    and motif[0] in ("A", "C", "G", "U")
    and motif[1] in ("A", "C", "G", "U")
)

data = Buffer.load("input/allpdb-rna-aareduce.mixed").deserialize("mixed")
coor = np.stack((data[1]["x"], data[1]["y"], data[1]["z"]), axis=1)

nucstart = np.where(data[1]["name"] == b"P")[0]
seq0 = data[1]["resname"][nucstart]
seq00 = seq0.tobytes().decode()
seq_c1, seq_c2 = seq00[::3], seq00[1::3]
seq = "".join([c2 if c1 == "R" else "X" for c1, c2 in zip(seq_c1, seq_c2)])

segstart = np.array(sorted([v[0] for v in data[0].values()]))
nucseg = np.searchsorted(segstart, nucstart, side="right") - 1
all_segnames = list(data[0].keys())
nucsegnames = [all_segnames[i] for i in nucseg]

is_nuc1 = np.array([c == motif[0] for c in seq], bool)
is_nuc2 = np.array([c == motif[1] for c in seq], bool)
same_seg = np.equal(nucseg[:-1], nucseg[1:])

is_dinuc = np.where(is_nuc1[:-1] & is_nuc2[1:] & same_seg)[0]

pos_ori = [nucsegnames[pos][:4] for pos in is_dinuc]

lib = np.load(f"library/library/dinuc-{motif}-0.5.npy")
lib_ori = [
    l.split()[0]
    for l in open(f"library/library/dinuc-{motif}-0.5-replacement.txt").readlines()[1:]
]
assert len(lib_ori) == len(lib)
lib_replacement = np.load(f"library/library/dinuc-{motif}-0.5-replacement.npy")
assert len(lib_replacement) == len(lib)

lib_ext = np.load(f"library/library/dinuc-{motif}-0.5-extension.npy")
lib_ori_ext = [
    l.strip() for l in open(f"library/library/dinuc-{motif}-0.5-extension.origin.txt")
]
assert len(lib_ori_ext) == len(lib_ext)

from nefertiti.functions.superimpose import superimpose_array

best_conf = np.empty(len(is_dinuc), np.uint16)
best_rmsd = np.empty(len(is_dinuc), np.float32)
for posnr, pos in enumerate(tqdm(is_dinuc, desc="Primary library")):
    start, end = nucstart[pos : pos + 3 : 2]
    ### assert end - start == lib.shape[1], (pos, end - start)
    c = coor[start:end]
    _, rmsd = superimpose_array(lib, c)
    best = rmsd.argmin()
    best_conf[posnr] = best
    best_rmsd[posnr] = rmsd[best]

best_ori = [lib_ori[best] for best in best_conf]
same_ori = np.array([ori1 == ori2 for ori1, ori2 in zip(best_ori, pos_ori)])

replacement_mask = np.zeros(len(is_dinuc), bool)
repl_posnr = np.where(same_ori)[0]
for posnr in tqdm(repl_posnr, desc="Replacement library"):
    curr_ori = pos_ori[posnr]
    curr_lib = lib.copy()
    ori_mask = [curr_lib_ori == curr_ori for curr_lib_ori in lib_ori]
    curr_lib[ori_mask] = lib_replacement[ori_mask]

    pos = is_dinuc[posnr]
    start, end = nucstart[pos : pos + 3 : 2]
    assert end - start == lib.shape[1], (pos, end - start)
    c = coor[start:end]
    _, rmsd = superimpose_array(curr_lib, c)
    best = rmsd.argmin()
    best_conf[posnr] = best
    best_rmsd[posnr] = rmsd[best]
    if ori_mask[best]:  # the best fit comes from the replacement library
        replacement_mask[posnr] = True

print("Replacement coordinates from primary fit:", replacement_mask.sum())

lib_offset = len(lib)
poor_fit_posnr = np.where(best_rmsd > 0.5)[0]
best_conf_ext = np.empty(len(poor_fit_posnr), np.uint16)
best_rmsd_ext = np.empty(len(poor_fit_posnr), np.float32)
lib_ori_ext = np.array(lib_ori_ext)
for posnr_ext, posnr in enumerate(tqdm(poor_fit_posnr, desc="Extension library")):
    pos = is_dinuc[posnr]
    start, end = nucstart[pos : pos + 3 : 2]
    ### assert end - start == lib.shape[1], (pos, end - start)

    curr_ori = pos_ori[posnr]
    ori_mask = np.equal(lib_ori_ext, curr_ori)

    c = coor[start:end]
    _, rmsd = superimpose_array(lib_ext, c)
    rmsd[ori_mask] = np.inf  # ignore fits to same-origin fragments

    best = rmsd.argmin()
    best_conf_ext[posnr_ext] = best
    best_rmsd_ext[posnr_ext] = rmsd[best]

improv_mask = best_rmsd_ext < best_rmsd[poor_fit_posnr]
posnr_improv = poor_fit_posnr[improv_mask]
best_conf[posnr_improv] = best_conf_ext[improv_mask] + lib_offset
best_rmsd[posnr_improv] = best_rmsd_ext[improv_mask]

print("Improved fit from extension library:", improv_mask.sum())

with open(f"allpdb-rna-fit-dinuc-{motif}.txt", "w") as f:
    for n, pos in enumerate(is_dinuc):
        conf = best_conf[n]
        rmsd = best_rmsd[n]
        replacement = int(replacement_mask[n])
        print(f"{pos+1} {conf+1} {rmsd:.5f} {replacement}", file=f)

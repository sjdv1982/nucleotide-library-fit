from seamless import Buffer
from tqdm import tqdm

import numpy as np

import sys

from nefertiti.functions.superimpose import superimpose

motif = sys.argv[1]
assert (
    len(motif) == 2
    and motif[0] in ("A", "C", "G", "U")
    and motif[1] in ("A", "C", "G", "U")
)

data = Buffer.load("input/allpdb-rna-aareduce.mixed").deserialize("mixed")
coor = np.stack((data[1]["x"], data[1]["y"], data[1]["z"]), axis=1)
nucstart = np.where(data[1]["name"] == b"P")[0]

fits = np.loadtxt(f"allpdb-rna-fit-dinuc-{motif}.txt")
fit_pos = (fits[:, 0] - 1).astype(np.uint)
fit_conf = (fits[:, 1] - 1).astype(np.uint)
fit_rmsd = fits[:, 2]
fit_is_replacement = fits[:, 3].astype(bool)

lib0 = np.load(f"library/library/dinuc-{motif}-0.5.npy")
lib_replacement = np.load(f"library/library/dinuc-{motif}-0.5-replacement.npy")
assert len(lib_replacement) == len(lib0)
lib_ext = np.load(f"library/library/dinuc-{motif}-0.5-extension.npy")

lib_offset = len(lib0)
lib = np.concatenate((lib0, lib_ext))

confs = lib[fit_conf]
to_replace = np.where(fit_is_replacement)[0]
inds = fit_conf[to_replace]
confs[to_replace] = lib_replacement[
    inds
]  # TODO: Re-run allpdb-rna-fit-dinuc.py when switching to real data

all_ermsd = []

e_ind = {
    # C2,C4,C6:
    "A": (12, 10, 14),
    "C": (11, 14, 10),
    "G": (12, 10, 15),
    "U": (11, 14, 10),
}
e_inds = np.concatenate((e_ind[motif[0]], e_ind[motif[1]])) - 1


e_rmsds = []
sqrt_natoms = np.sqrt(len(lib[0]))
sqrt_ne = np.sqrt(len(e_inds))
for conf, conformer, pos, rmsd in tqdm(list(zip(confs, fit_conf, fit_pos, fit_rmsd))):
    pos = int(pos)
    start, end = nucstart[pos : pos + 3 : 2]
    c = coor[start:end]
    c = c - c.mean(axis=0)
    rotmat, rmsd0 = superimpose(conf, c)
    assert abs(rmsd - rmsd0) < 0.01
    rconf = conf.dot(rotmat)
    rmsd00 = np.sqrt((((rconf - c) ** 2).sum())) / sqrt_natoms
    assert abs(rmsd - rmsd00) < 0.01
    e_rconf = rconf[e_inds]
    e_c = c[e_inds]
    e_rmsd = np.sqrt((((e_rconf - e_c) ** 2).sum())) / sqrt_ne
    e_rmsds.append(e_rmsd)

np.savetxt(f"allpdb-rna-fit-dinuc-{motif}.ermsd", e_rmsds, fmt="%.6f")

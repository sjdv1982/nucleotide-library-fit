import numpy as np
import sys

from tqdm import tqdm
from nefertiti.functions.superimpose import superimpose_array

pairseq = sys.argv[1]
assert (
    len(pairseq) == 3
    and pairseq[0] in ("A", "C")
    and pairseq[1] in ("A", "C")
    and pairseq[2] in ("A", "C")
), pairseq
baselen = {"A": 22, "C": 20, "G": 23, "U": 20}

dinuc1 = pairseq[:2]
dinuc2 = pairseq[-2:]
print(dinuc1, dinuc2)

libs = []
for dinuc in (dinuc1, dinuc2):
    lib0 = np.load(f"library/library/dinuc-{dinuc}-0.5.npy")
    lib_ext = np.load(f"library/library/dinuc-{dinuc}-0.5-extension.npy")
    lib = np.concatenate((lib0, lib_ext)).astype(float)
    assert lib.shape[1] == baselen[dinuc[0]] + baselen[dinuc[1]]
    libs.append(lib)

lib1, lib2 = libs
common = baselen[pairseq[1]]
lib1 = lib1[:, -common:]
lib2 = lib2[:, :common]

e_ind = {
    # C2,C4,C6:
    "A": (12, 10, 14),
    "C": (11, 14, 10),
    "G": (12, 10, 15),
    "U": (11, 14, 10),
}
lib1c = lib1 - lib1.mean(axis=1)[:, None, :]
lib2c = lib2 - lib2.mean(axis=1)[:, None, :]

e_inds = np.array(e_ind[pairseq[1]]) - 1
lib1e = lib1c[:, e_inds]
lib2e = lib2c[:, e_inds]

ecrmsd_matrix = np.empty((len(lib1), len(lib2)))

sqrt_natoms = np.sqrt(len(e_inds))
for confnr, conf in enumerate(tqdm(lib1c)):
    rotmats, rmsd = superimpose_array(lib2c, conf)
    erlib2 = np.einsum("ijk,ikl->ijl", lib2e, rotmats)
    econf = lib1e[confnr]
    d = erlib2 - econf[None, :, :]
    sd = np.einsum("ijk,ijk->i", d, d)
    rmsd2 = np.sqrt(sd) / sqrt_natoms
    ecrmsd_matrix[confnr] = rmsd2

np.save(f"ecrmsd_matrix_{pairseq}.npy", ecrmsd_matrix)

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

crmsd_matrix = np.empty((len(lib1), len(lib2)))

for confnr, conf in enumerate(tqdm(lib1)):
    _, rmsd = superimpose_array(lib2, conf)
    crmsd_matrix[confnr] = rmsd

np.save(f"crmsd_matrix_{pairseq}.npy", crmsd_matrix)

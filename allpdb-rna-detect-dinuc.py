from seamless import Buffer
from tqdm import tqdm

import numpy as np

import sys

BOND_THRESHOLD = 1.7
DISCONNECT_THRESHOLD = 3.0

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

is_dinuc0 = np.where(is_nuc1[:-1] & is_nuc2[1:] & same_seg)[0]
is_dinuc = []
tmpl = np.load(f"templates/{motif}-ppdb.npy")

tmpl_coor = np.stack((tmpl["x"], tmpl["y"], tmpl["z"]), axis=1)
d = tmpl_coor[:, None, :] - tmpl_coor[None, :, :]
tmpl_dist = np.sqrt((d * d).sum(axis=2))
tmpl_bondmat = tmpl_dist < BOND_THRESHOLD
p1, p2 = np.where(tmpl_bondmat)
bonds = []
for pp1, pp2 in zip(p1, p2):
    if pp1 >= pp2:
        continue

    a1 = tmpl[pp1]
    a2 = tmpl[pp2]

    if a1["resid"] != a2["resid"]:
        if a2["name"] != b"P":
            # print("REJECT")
            continue
    bonds.append((pp1, pp2))

p1 = [pp1 for pp1, _ in bonds]
p2 = [pp2 for _, pp2 in bonds]

for pos in is_dinuc0:
    try:
        start, end = nucstart[pos : pos + 3 : 2]
    except ValueError:
        # print("ERR", pos)
        continue
    if end - start != len(tmpl):
        continue

    c = coor[start:end]
    frag_p1 = c[p1]
    frag_p2 = c[p2]
    d = frag_p1 - frag_p2
    dis = np.sqrt((d * d).sum(axis=1))
    maxdis = dis.max()
    if maxdis > DISCONNECT_THRESHOLD:
        continue

    is_dinuc.append(pos)
is_dinuc = np.array(is_dinuc)

print(len(is_dinuc))
np.save(f"allpdb-rna-detect-dinuc-{motif}.npy", is_dinuc)

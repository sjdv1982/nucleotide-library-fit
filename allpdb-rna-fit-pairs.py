from seamless import Buffer

import numpy as np

import sys

motif = sys.argv[1]
assert (
    len(motif) == 3
    and motif[0] in ("A", "C", "G", "U")
    and motif[1] in ("A", "C", "G", "U")
    and motif[2] in ("A", "C", "G", "U")
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

assert len(seq) == len(nucseg)

motif2 = motif.replace("G", "A").replace("U", "C")
crmsd_matrix = np.load(f"crmsd_matrix_{motif2}.npy")
ecrmsd_matrix = np.load(f"ecrmsd_matrix_{motif2}.npy")
fits1 = np.load(f"allpdb-rna-fit-dinuc-{motif[:2]}-rotamers.npy")
fits1 = fits1[fits1["conformer"] >= 0]
fits2 = np.load(f"allpdb-rna-fit-dinuc-{motif[-2:]}-rotamers.npy")
fits2 = fits2[fits2["conformer"] >= 0]

baselen = {"A": 22, "C": 20, "G": 23, "U": 20}


def get_confs(fits, submotif):
    lib0 = np.load(f"library/library/dinuc-{submotif}-0.5.npy")

    lib_ext = np.load(f"library/library/dinuc-{submotif}-0.5-extension.npy")

    lib_offset = len(lib0)
    lib = np.concatenate((lib0, lib_ext)).astype(float)
    # Re-center the library; this normally necessary for mutated libraries
    lib -= lib.mean(axis=1)[:, None, :]

    lib_replacement = np.load(
        f"library/library/dinuc-{submotif}-0.5-replacement.npy"
    ).astype(float)
    # Re-center the library
    lib_replacement -= lib_replacement.mean(axis=1)[:, None, :]
    assert len(lib_replacement) == len(lib0)

    confs = lib[fits["conformer"]]
    to_replace = np.where(fits["replacement_conformer"])[0]
    inds = fits["conformer"][to_replace]
    confs[to_replace] = lib_replacement[inds]

    return confs


confs1 = get_confs(fits1, motif[:2])
# print(confs1.shape)
confs1 = confs1[:, -baselen[motif[1]] :]
# print(confs1.shape)

confs2 = get_confs(fits2, motif[-2:])
# print(confs2.shape)
confs2 = confs2[:, : baselen[motif[1]]]
# print(confs2.shape)

e_ind = {
    # C2,C4,C6:
    "A": (12, 10, 14),
    "C": (11, 14, 10),
    "G": (12, 10, 15),
    "U": (11, 14, 10),
}

e_inds = np.array(e_ind[motif[1]]) - 1

pair_dtype = np.dtype(
    [
        ("frag_index", np.uint32),
        ("cRMSD", np.float32),
        ("e-cRMSD", np.float32),
        ("ovRMSD", np.float32),
        ("e-ovRMSD", np.float32),
    ],
    align=True,
)
npairs = 0
pairs = np.empty(len(fits1), pair_dtype)

all_pos1 = fits1["frag_index"]
if all_pos1[-1] == len(seq) - 1:
    all_pos1 = all_pos1[:-1]
all_pos2 = {v: i for i, v in enumerate(fits2["frag_index"])}

for posnr1, pos1 in enumerate(all_pos1):
    pos2 = pos1 + 1
    if pos2 in all_pos2 and nucseg[pos1] == nucseg[pos2]:
        posnr2 = all_pos2[pos2]
        f1, f2 = fits1[posnr1], fits2[posnr2]
        if f1["conformer"] == -1 or f2["conformer"] == -1:
            continue
        pair = pairs[npairs]
        npairs += 1
        pair["frag_index"] = pos1

        cRMSD, ecRMSD = -1, -1
        cRMSD = crmsd_matrix[f1["conformer"], f2["conformer"]]
        ecRMSD = ecrmsd_matrix[f1["conformer"], f2["conformer"]]
        pair["cRMSD"] = cRMSD
        pair["e-cRMSD"] = ecRMSD

        conf1 = confs1[posnr1]
        conf2 = confs2[posnr2]
        pose1 = conf1.dot(f1["rotmat"]) + f1["offset"]
        pose2 = conf2.dot(f2["rotmat"]) + f2["offset"]
        d = pose1 - pose2
        ovRMSD = ((d * d).sum() / len(pose1)) ** 0.5
        pair["ovRMSD"] = ovRMSD

        e_d = d[e_inds]
        e_ovRMSD = ((e_d * e_d).sum() / len(e_inds)) ** 0.5
        pair["e-ovRMSD"] = e_ovRMSD

pairs = pairs[:npairs]
np.save(f"allpdb-rna-fit-dinuc-{motif}-pairs.npy", pairs)

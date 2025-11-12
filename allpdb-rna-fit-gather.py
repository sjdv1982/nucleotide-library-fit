from tqdm import tqdm
from seamless import Buffer
import numpy as np

motifs = []
pair_motifs = []
for m1 in ("A", "C", "G", "U"):
    for m2 in ("A", "C", "G", "U"):
        motifs.append(m1 + m2)
        for m3 in ("A", "C", "G", "U"):
            pair_motifs.append(m1 + m2 + m3)

data = Buffer.load("input/allpdb-rna-aareduce.mixed").deserialize("mixed")
coor = np.stack((data[1]["x"], data[1]["y"], data[1]["z"]), axis=1)
nucstart = np.where(data[1]["name"] == b"P")[0]
nucresids = data[1]["resid"][nucstart]

seq0 = data[1]["resname"][nucstart]
seq00 = seq0.tobytes().decode()
seq_c1, seq_c2 = seq00[::3], seq00[1::3]
seq = "".join([c2 if c1 == "R" else "X" for c1, c2 in zip(seq_c1, seq_c2)])
assert len(seq) == len(nucresids)

segstart = np.array(sorted([v[0] for v in data[0].values()]))
nucseg = np.searchsorted(segstart, nucstart, side="right") - 1
all_segnames = list(data[0].keys())
nucsegnames = [all_segnames[i] for i in nucseg]

nucfragments = []
curr_seg = None
curr_ind = 0
for seg in nucseg:
    if seg != curr_seg:
        curr_seg = seg
        curr_ind = 0
    curr_ind += 1
    nucfragments.append(curr_ind)
nucfragments = np.array(nucfragments)

maxnucseglen = max([len(l) for l in nucsegnames])

fit_dtype = np.dtype(
    [
        ("pdb", "S" + str(maxnucseglen)),
        ("fragment", np.uint16),
        ("resid", np.uint16),
        ("sequence", "S2"),
        ("conformer", np.int16),
        ("replacement_conformer", bool),
        ("conf_rmsd", np.float32),
        ("e-conf_rmsd", np.float32),
        ("rotamer", np.uint32),
        ("rotmat", float, (3, 3)),
        ("offset", np.float32, 3),
        ("drmsd", np.float32),
    ],
    align=True,
)

print("START")

frag_indices = []

fits = np.empty(len(seq), dtype=fit_dtype)
for motif in tqdm(motifs):
    cfits = np.load(f"allpdb-rna-fit-dinuc-{motif}-rotamers.npy")
    c_ermsds = np.loadtxt(f"allpdb-rna-fit-dinuc-{motif}.ermsd")
    assert len(cfits) == len(c_ermsds), motif
    for fit0, ermsd in zip(cfits, c_ermsds):
        pos = fit0["frag_index"]
        conf = fit0["conformer"]
        if conf == -1 or nucresids[pos + 1] != nucresids[pos] + 1:
            continue

        frag_indices.append(pos)
        fit = fits[pos]
        fit["pdb"] = nucsegnames[pos]
        fit["fragment"] = nucfragments[pos]
        fit["resid"] = nucresids[pos]
        fit["sequence"] = motif
        fit["conformer"] = conf
        for field in (
            "replacement_conformer",
            "conf_rmsd",
            "rotamer",
            "rotmat",
            "offset",
            "drmsd",
        ):
            fit[field] = fit0[field]
        fit["e-conf_rmsd"] = ermsd

fits = fits[frag_indices]
order = np.argsort(frag_indices)
frag_indices = np.array(frag_indices)[order]
fits = fits[order]
np.save("allpdb-rna-fit.npy", fits)

pair_dtype = np.dtype(
    [
        ("index", np.uint32),
        ("cRMSD", np.float32),
        ("e-cRMSD", np.float32),
        ("ovRMSD", np.float32),
        ("e-ovRMSD", np.float32),
    ],
    align=True,
)

rev_frag_indices = {v: i for i, v in enumerate(frag_indices)}

pairs = np.empty(len(fits), pair_dtype)
npairs = 0
for motif in tqdm(pair_motifs):
    cpairs = np.load(f"allpdb-rna-fit-dinuc-{motif}-pairs.npy")
    for cpair in cpairs:
        frag_index = cpair["frag_index"]
        i1 = rev_frag_indices.get(frag_index)
        i2 = rev_frag_indices.get(frag_index + 1)
        if i1 is None or i2 is None:
            continue
        assert i2 == i1 + 1, (frag_index, i1, i2)
        f1, f2 = fits[i1 : i1 + 2]
        seq1, seq2 = f1["sequence"], f2["sequence"]
        assert seq1[1] == seq2[0]
        assert (seq1 + seq2[1:]).decode() == motif
        pair = pairs[npairs]
        pair["index"] = i1 + 1
        for field in ("cRMSD", "e-cRMSD", "ovRMSD", "e-ovRMSD"):
            pair[field] = cpair[field]
        npairs += 1

pairs = pairs[:npairs]
order = np.argsort(pairs["index"])
pairs = pairs[order]
np.save("allpdb-rna-fit-pairs.npy", pairs)

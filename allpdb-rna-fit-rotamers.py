from seamless import Buffer
from tqdm import tqdm

import numpy as np

import sys

from nefertiti.functions.superimpose import superimpose
from scipy.spatial.transform import Rotation

GRIDSPACING = np.sqrt(3) / 3

fit_dtype = np.dtype(
    [
        ("frag_index", np.uint32),
        ("conformer", np.int16),
        ("replacement_conformer", bool),
        ("conf_rmsd", np.float32),
        ("rotamer", np.uint32),
        ("rotmat", float, (3, 3)),
        ("offset", np.float32, 3),
        ("drmsd", np.float32),
    ],
    align=True,
)

motif = sys.argv[1]
assert (
    len(motif) == 2
    and motif[0] in ("A", "C", "G", "U")
    and motif[1] in ("A", "C", "G", "U")
)

data = Buffer.load("input/allpdb-rna-aareduce.mixed").deserialize("mixed")
coor = np.stack((data[1]["x"], data[1]["y"], data[1]["z"]), axis=1)
nucstart = np.where(data[1]["name"] == b"P")[0]

fits0 = np.loadtxt(f"allpdb-rna-fit-dinuc-{motif}.txt")
fit_pos = (fits0[:, 0] - 1).astype(np.uint)
fit_conf = (fits0[:, 1] - 1).astype(int)
print(fit_conf.min())
fit_rmsd = fits0[:, 2]
fit_is_replacement = fits0[:, 3].astype(bool)
fits = np.empty(len(fits0), fit_dtype)
fits["frag_index"] = fit_pos
fits["conformer"] = fit_conf
fits["replacement_conformer"] = fit_is_replacement
fits["conf_rmsd"] = fit_rmsd

lib0 = np.load(f"library/library/dinuc-{motif}-0.5.npy")

lib_ext = np.load(f"library/library/dinuc-{motif}-0.5-extension.npy")

lib_offset = len(lib0)
lib = np.concatenate((lib0, lib_ext)).astype(float)
# Re-center the library; this normally necessary for mutated libraries
lib -= lib.mean(axis=1)[:, None, :]


lib_replacement = np.load(f"library/library/dinuc-{motif}-0.5-replacement.npy").astype(
    float
)
# Re-center the library
lib_replacement -= lib_replacement.mean(axis=1)[:, None, :]
assert len(lib_replacement) == len(lib0)

confs = lib[fit_conf]
to_replace = np.where(fit_is_replacement)[0]
inds = fit_conf[to_replace]
confs[to_replace] = lib_replacement[inds]

motif2 = motif.replace("G", "A").replace("U", "C")
rota_index = np.load(f"input/rotaconformers/dinuc-{motif2}-0.5.index.npy")
rota = np.load(f"input/rotaconformers/dinuc-{motif2}-0.5.npy")
rota_index = np.concatenate(
    (
        rota_index,
        np.load(f"input/rotaconformers/dinuc-{motif2}-0.5-extension.index.npy")
        + len(rota),
    )
)
rota = np.concatenate(
    (rota, np.load(f"input/rotaconformers/dinuc-{motif2}-0.5-extension.npy"))
)


def get_rotamers(conformer):
    conformer = int(conformer)
    last = rota_index[conformer]
    if conformer == 0:
        first = 0
    else:
        first = rota_index[conformer - 1]
    return Rotation.from_rotvec(rota[first:last]).as_matrix()


sqrt_natoms = np.sqrt(len(lib[0]))
for c_fit, conf, conformer, pos, rmsd in tqdm(
    list(zip(fits, confs, fit_conf, fit_pos, fit_rmsd))
):
    pos = int(pos)
    start, end = nucstart[pos : pos + 3 : 2]
    if end - start != len(conf):
        continue

    c = coor[start:end]
    com = c.mean(axis=0)
    c_offset = np.round(com / GRIDSPACING) * GRIDSPACING
    dcom = com - c_offset
    dmsd_trans = (dcom * dcom).sum()
    c = c - com
    rotmat, rmsd0 = superimpose(conf, c)
    assert abs(rmsd - rmsd0) < 0.01
    rconf = conf.dot(rotmat)
    rmsd00 = np.sqrt((((rconf - c) ** 2).sum())) / sqrt_natoms
    assert abs(rmsd - rmsd00) < 0.01, (rmsd, rmsd00, rconf.mean(axis=0), c.mean(axis=0))

    rotamers = get_rotamers(conformer)

    rotamer_coors = np.einsum("jk,ikl->ijl", conf, rotamers)
    d = rotamer_coors - c
    sd = np.einsum("ijk,ijk->i", d, d)
    rotamer_rmsd = np.sqrt(sd) / sqrt_natoms
    c_rotamer = rotamer_rmsd.argmin()
    c_rotmat = rotamers[c_rotamer]
    dmsd_rot = rotamer_rmsd[c_rotamer]
    c_drmsd = np.sqrt(dmsd_rot + dmsd_trans)
    c_fit["rotamer"] = c_rotamer
    c_fit["rotmat"] = c_rotmat
    c_fit["offset"] = c_offset
    c_fit["drmsd"] = c_drmsd


np.save(f"allpdb-rna-fit-dinuc-{motif}-rotamers.npy", fits)

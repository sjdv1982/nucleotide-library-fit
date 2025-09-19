import numpy as np
from nefertiti.functions.superimpose import superimpose
from nefertiti.functions.write_pdb import write_pdb, write_multi_pdb
from nefertiti.functions.parse_pdb import atomic_dtype

pair_motifs = ("AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC")
motifs = ("AA", "AC", "CA", "CC")
baselen = {"A": 22, "C": 20, "G": 23, "U": 20}

templates = {}
template_masks = {}
for m in motifs + pair_motifs + ("A", "C"):
    tmplf = f"templates/{m}-ppdb.npy"
    templates[m] = np.load(tmplf)

lib = {}
for motif in motifs:
    libf = f"library/library/dinuc-{motif}-0.5.npy"
    lib[motif] = np.load(libf)

for pair_motif in pair_motifs:
    m1 = pair_motif[:2]
    m2 = pair_motif[-2:]
    common = m1[1]
    sup_lib1 = lib[m1][:, -baselen[common] :]
    sup_lib2 = lib[m2][:, : baselen[common]]
    fname = f"pairstats-{pair_motif}-antipairs.txt"
    ###crmsd = np.load(f"crmsd_matrix_{pair_motif}.npy")

    antipairs_nonmerged = []
    antipairs = []

    nucpos1 = baselen[pair_motif[0]]
    nucpos2 = nucpos1 + baselen[pair_motif[1]]

    with open(fname) as f:
        for l in f.readlines()[1:]:
            ll = l.split()
            conf1, conf2 = int(ll[0]), int(ll[1])
            sup_conf1 = sup_lib1[conf1 - 1]
            sup_conf2 = sup_lib2[conf2 - 1]
            rotmat, rmsd = superimpose(sup_conf2, sup_conf1)

            write_conf1 = lib[m1][conf1 - 1]
            write_conf2_0 = lib[m2][conf2 - 1]
            write_conf2 = write_conf2_0.dot(rotmat)
            write_conf2_common = write_conf2[: baselen[common]]
            offset = sup_conf1.mean(axis=0) - write_conf2_common.mean(axis=0)
            write_conf2 += offset

            """
            dif = write_conf2_common - sup_conf1
            print(
                np.sqrt((dif * dif).sum() / len(sup_conf1)), crmsd[conf1 - 1, conf2 - 1]
            )
            """

            tmpl_conf1 = templates[m1].copy()
            tmpl_conf1["x"], tmpl_conf1["y"], tmpl_conf1["z"] = write_conf1.T

            tmpl_conf2 = templates[m2].copy()
            tmpl_conf2["x"], tmpl_conf2["y"], tmpl_conf2["z"] = write_conf2.T
            tmpl_conf2["resid"][: baselen[common]] = 100
            tmpl_conf2["resid"][baselen[common] :] = 101

            antipair_nonmerged = np.concatenate((tmpl_conf1, tmpl_conf2)).astype(
                atomic_dtype
            )
            antipairs_nonmerged.append(antipair_nonmerged)

            tmpl = templates[pair_motif].copy()

            tmpl_nuc = tmpl[:nucpos1]
            conf_nuc = write_conf1[:nucpos1]
            tmpl_nuc["x"], tmpl_nuc["y"], tmpl_nuc["z"] = conf_nuc.T

            tmpl_nuc = tmpl[nucpos1:nucpos2]
            conf_nuc = (write_conf2_common + sup_conf1) / 2
            tmpl_nuc["x"], tmpl_nuc["y"], tmpl_nuc["z"] = conf_nuc.T

            tmpl_nuc = tmpl[nucpos2:]
            conf_nuc = write_conf2[-len(tmpl_nuc) :]
            tmpl_nuc["x"], tmpl_nuc["y"], tmpl_nuc["z"] = conf_nuc.T

            antipairs.append(tmpl)

    antipairs_nonmerged = np.stack(antipairs_nonmerged).astype(
        antipairs_nonmerged[0].dtype
    )
    pdbtxt = write_multi_pdb(antipairs_nonmerged)
    with open(f"antipairs-{pair_motif}-nonmerged.pdb", "w") as f:
        f.write(pdbtxt)

    antipairs = np.stack(antipairs).astype(antipairs[0].dtype)
    pdbtxt = write_multi_pdb(antipairs)
    with open(f"antipairs-{pair_motif}.pdb", "w") as f:
        f.write(pdbtxt)

    refe = templates[pair_motif[1]]
    refe = np.stack((refe["x"], refe["y"], refe["z"]), axis=1)

    for antipair in antipairs:
        coor = np.stack((antipair["x"], antipair["y"], antipair["z"]), axis=1)
        coor_fit = coor[nucpos1:nucpos2]
        coor -= coor_fit.mean(axis=0)
        rotmat, _ = superimpose(coor_fit, refe)
        coor = coor.dot(rotmat)
        coor += refe.mean(axis=0)
        antipair["x"], antipair["y"], antipair["z"] = coor.T

    pdbtxt = write_multi_pdb(antipairs)
    with open(f"antipairs-{pair_motif}-aligned.pdb", "w") as f:
        f.write(pdbtxt)

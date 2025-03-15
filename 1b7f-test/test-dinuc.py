import os
import sys
import numpy as np
import itertools
from nefertiti.functions.write_pdb import write_pdb
from nefertiti.functions.superimpose import superimpose


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


bases = ("A", "C", "G", "U")
dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=2)]

template_pdbs = {}
for seq in dinuc_sequences:
    filename = f"../templates/{seq}-template-ppdb.npy"
    template = np.load(filename)
    template_pdbs[seq] = template

conformers = {}
for seq in dinuc_sequences:
    conformers0 = []
    for filename in (
        f"../library/dinuc-{seq}-0.5.npy",
        f"../library/dinuc-{seq}-0.5-extension.npy",
    ):
        conformer = np.load(filename)
        if conformer.dtype not in (np.float32, np.float64):
            err(f"Conformer file '{filename}' does not contain an array of numbers")
        if conformer.ndim != 3:
            err(f"Conformer file '{filename}' does not contain a 3D coordinate array")
        conformers0.append(conformer.astype(float))
    conformers[seq] = np.concatenate(conformers0)
    np.save(f"lib-{seq}.npy", conformers[seq])
    del conformers0

pre_masks = {}
post_masks = {}
for seq in dinuc_sequences:
    template = template_pdbs[seq]
    pre_masks[seq] = template["resid"] == 2
    post_masks[seq] = template["resid"] == 1

FIT_THRESHOLD = 1.0

best_fit = np.load("dinuc-best-fit.npy")
for n1, r1 in enumerate(best_fit[:-1]):
    r2 = best_fit[n1 + 1]
    if r1["rmsd"] > FIT_THRESHOLD:
        continue
    if r2["rmsd"] > FIT_THRESHOLD:
        continue

    seq1 = r1["sequence"].decode()
    fragnr = r1["first_resid"]
    lib1 = conformers[seq1]
    coors1_00 = lib1[r1["conformer"]]
    coors1_0 = np.ones((len(coors1_00), 4))
    coors1_0[:, :3] = coors1_00
    coors1 = coors1_0.dot(r1["matrix"])[:, :3]

    pdb = template_pdbs[seq1].copy()
    pdb["x"], pdb["y"], pdb["z"] = coors1[:, 0], coors1[:, 1], coors1[:, 2]
    pdb_txt = write_pdb(pdb, ter=True)
    with open(f"frag-{r1['first_resid']}.pdb", "w") as f:
        f.write(pdb_txt)

    ccoor1 = coors1[pre_masks[seq1]]
    ccom1 = ccoor1.mean(axis=0)
    seq2 = r2["sequence"].decode()
    lib2 = conformers[seq2]

    outf = f"frag-{fragnr}-crmsd.txt"
    poses = []
    rmsds = []
    for i, conf in enumerate(lib2):
        ccoor2 = conf[post_masks[seq2]]
        ccom2 = ccoor2.mean(axis=0)
        rotmat, rmsd = superimpose(ccoor2, ccoor1)
        rmsds.append(rmsd)
        pose = np.empty((4, 4))
        pose[:3, :3] = rotmat
        pose[3, :3] = ccom1 - ccom2.dot(rotmat)
        pose[3, 3] = 1
        poses.append(pose)

    # Test that superposition on the common conformer approximates the superposed-onto-bound coordinates
    cconf = lib2[r2["conformer"]]
    cconfx = np.ones((len(cconf), 4))
    cconfx[:, :3] = cconf
    f1 = cconfx.dot(r2["matrix"])[:, :3]
    f2 = cconfx.dot(poses[r2["conformer"]])[:, :3]
    d = f1 - f2
    r = np.sqrt((d * d).sum() / len(f1))
    if r > 1:
        continue

    print(fragnr)

    outf = f"frag-{fragnr}-true.txt"
    with open(outf, "w") as f:
        print(r2["conformer"] + 1, file=f)

    outf = f"frag-{fragnr}-crmsd.txt"
    with open(outf, "w") as f:
        for rmsd in rmsds:
            print(f"{rmsd:.3f}", file=f)

    poses = np.array(poses)
    np.save(f"frag-{fragnr}-poses.npy", poses)
    np.save(f"frag-{fragnr}-conf.npy", np.arange(len(lib2)).astype(np.uint32) + 1)
    try:
        os.remove(f"frag-{fragnr}-lib.npy")
    except FileNotFoundError:
        pass
    try:
        os.remove(f"frag-{fragnr}-atomtypes.npy")
    except FileNotFoundError:
        pass
    os.symlink(f"lib-{seq2}-reduced.npy", f"frag-{fragnr}-lib.npy")
    os.symlink(
        f"../templates-reduced/{seq2}-template-atomtypes.npy",
        f"frag-{fragnr}-atomtypes.npy",
    )

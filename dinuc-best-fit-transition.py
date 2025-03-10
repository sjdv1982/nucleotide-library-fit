import itertools
import sys
import numpy as np
from seamless import Buffer
from crocodile.trinuc.best_fit import best_fit
from nefertiti.functions.parse_mmcif import atomic_dtype
from nefertiti.functions.superimpose import superimpose


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


bases = ("A", "C", "G", "U")
dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=2)]

template_pdbs = {}
for seq in dinuc_sequences:
    filename = f"templates/{seq}-template-ppdb.npy"
    template = np.load(filename)
    if template.dtype != atomic_dtype:
        err(f"Template '{filename}' does not contain a parsed PDB")
    template_pdbs[seq] = template

pre_masks = {}
post_masks = {}
for seq in dinuc_sequences:
    template = template_pdbs[seq]
    pre_masks[seq] = template["resid"] == 2
    post_masks[seq] = template["resid"] == 1
    print(seq, post_masks[seq].sum(), pre_masks[seq].sum())


conformers = {}
for seq in dinuc_sequences:
    conformers0 = []
    for filename in (
        f"library/dinuc-{seq}-0.5.npy",
        f"library/dinuc-{seq}-0.5-extension.npy",
    ):
        conformer = np.load(filename)
        if conformer.dtype not in (np.float32, np.float64):
            err(f"Conformer file '{filename}' does not contain an array of numbers")
        if conformer.ndim != 3:
            err(f"Conformer file '{filename}' does not contain a 3D coordinate array")
        if conformer.shape[1] != len(template_pdbs[seq]):
            err(
                f"Sequence {seq}: conformer '{filename}' doesn't have the same number of atoms as the template"
            )
        conformers0.append(conformer.astype(float))
    conformers0 = np.concatenate(conformers0)
    conformers[seq] = np.ones(conformers0.shape[:2] + (4,))
    conformers[seq][:, :, :3] = conformers0
    del conformers0

dinuc_fits_index, dinuc_fits_data = Buffer.load(
    "allpdb-dinuc-fit-chunk-1.mixed"
).deserialize(
    "mixed"
)  ### TODO: all chunks

overlaps = []
for code, (start, length) in dinuc_fits_index.items():

    dinuc_fit = dinuc_fits_data[start : start + length]
    # to eliminate 0 A fits...
    best_dinuc_fit1 = best_fit(dinuc_fit, nth_best=1)
    best_dinuc_fit2 = best_fit(dinuc_fit, nth_best=2)
    """
    best_dinuc_fit = np.where(
        best_dinuc_fit1["rmsd"] > 0, best_dinuc_fit1, best_dinuc_fit2
    )
    """
    best_dinuc_fit = []
    for n in range(len(best_dinuc_fit1)):
        bf1, bf2 = best_dinuc_fit1[n], best_dinuc_fit2[n]
        try:
            assert bf1["sequence"] == bf2["sequence"], n
        except AssertionError:
            print("ERR", code, n)
            continue
        if bf1["rmsd"] == 0:
            bf = bf2
        else:
            bf = bf1
        best_dinuc_fit.append(bf)
    best_dinuc_fit = np.array(best_dinuc_fit)
    fitted = []
    last_f = None
    for f in best_dinuc_fit:
        if last_f is not None:
            try:
                assert f["first_resid"] == last_f["first_resid"] + 1
                assert f["sequence"][0] == last_f["sequence"][1]
            except AssertionError:
                print("ERR", code, f["first_resid"])
                fitted.append(None)
                last_f = f
                continue
        last_f = f
        conf = f["conformer"]
        struc0 = conformers[f["sequence"].decode()][conf]
        struc = struc0.dot(f["matrix"])
        fitted.append(struc[:, :3])

    for n in range(1, len(fitted)):
        last, next = fitted[n - 1], fitted[n]
        if last is None or next is None:
            continue
        last_seq, next_seq = (
            best_dinuc_fit[n - 1]["sequence"],
            best_dinuc_fit[n]["sequence"],
        )
        last = last[pre_masks[last_seq.decode()]]
        next = next[post_masks[next_seq.decode()]]
        overlap_rmsd = np.sqrt(((last - next) ** 2).sum() / len(last))
        last_rmsd, next_rmsd = best_dinuc_fit["rmsd"][n - 1], best_dinuc_fit["rmsd"][n]
        compat_rmsd = superimpose(last, next)[1]
        overlaps.append(
            (
                last_rmsd,
                next_rmsd,
                overlap_rmsd,
                compat_rmsd,
                best_dinuc_fit[n - 1]["conformer"],
                best_dinuc_fit[n]["conformer"],
            )
        )

overlaps = np.array(overlaps)
good_fit = np.max(overlaps[:, :2], axis=1) < 0.5
print(f"Both pairs fit < 0.5: {good_fit.mean()*100:.1f} %")
print("ovRMSD histo")
print(np.histogram(overlaps[good_fit, 2], bins=np.arange(0, 1.1, 0.1)))
print(
    np.histogram(overlaps[good_fit, 2], bins=np.arange(0, 1.1, 0.1))[0]
    / len(overlaps)
    * 100
)
print("cRMSD histo")
print(np.histogram(overlaps[good_fit, 3], bins=np.arange(0, 1.1, 0.1)))
print(
    np.histogram(overlaps[good_fit, 3], bins=np.arange(0, 1.1, 0.1))[0]
    / len(overlaps)
    * 100
)

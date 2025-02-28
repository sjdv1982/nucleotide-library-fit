import itertools
import os
import numpy as np
import sys
from seamless import Buffer
from dinuc_fit import dinuc_fit
from nefertiti.functions.parse_mmcif import atomic_dtype
from crocodile.nuc import map_resname


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


NCHUNKS = 100
chunk = int(sys.argv[1])

rna_segments = {}
for l in open("input/allpdb-segments.txt").readlines():
    code, start, length = l.split()
    start, length = int(start), int(length)
    if code not in rna_segments:
        rna_segments[code] = []
    rna_segments[code].append((start, length))

rna_struc_index, rna_strucs_data = Buffer.load(
    "input/allpdb-rna-aareduce.mixed"
).deserialize("mixed")

rna_codes = []
rna_strucs = []
for code in rna_struc_index:
    ###if not code.startswith("1b7f") and not code.startswith("1cvj"): continue ###
    start, length = rna_struc_index[code]
    struc = rna_strucs_data[start : start + length]
    segments = rna_segments[code]
    if (
        len(segments) == 1
        and segments[0][0] == 1
        and struc[0]["resid"] == 1
        and struc[-1]["resid"] == segments[0][1]
    ):
        rna_strucs.append(struc)
        rna_codes.append(code)
    else:
        for segnr, seg in enumerate(segments):
            start, length = seg
            mask = (struc["resid"] >= start) & (struc["resid"] <= start + length - 1)
            segstruc = struc[mask]
            segcode = code + "_" + str(segnr + 1)
            rna_strucs.append(segstruc)
            rna_codes.append(segcode)

bases = ("A", "C", "G", "U")
dinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=2)]

template_pdbs = {}
for seq in dinuc_sequences:
    filename = f"templates/{seq}-template-ppdb.npy"
    template = np.load(filename)
    if template.dtype != atomic_dtype:
        err(f"Template '{filename}' does not contain a parsed PDB")
    template_pdbs[seq] = template


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
    conformers[seq] = np.concatenate(conformers0)
    del conformers0

chunk_indices = [n for n in range(len(rna_codes)) if (n % NCHUNKS) == (chunk - 1)]
print(len(chunk_indices))
rna_codes = [rna_codes[n] for n in chunk_indices]
rna_strucs = [rna_strucs[n] for n in chunk_indices]

rmsd_margin = np.sqrt(
    3 * 0.25**2 + 0.5**2
)  # translational and rotational error for crocodile sampling
result = dinuc_fit(
    rna_strucs,
    rna_codes,
    template_pdbs=template_pdbs,
    dinuc_conformer_library=conformers,
    rmsd_margin=rmsd_margin,
    rmsd_soft_max=1.5,
)

# np.save(f"allpdb-overlap-rmsd-chunk-{chunk}.npy", result)
print(result[0])
buf = Buffer(result, celltype="mixed")
buf.save(f"allpdb-dinuc-fit-chunk-{chunk}.mixed")

import itertools
import json
import os
import numpy as np
import sys
from seamless import Buffer
from trinuc_fit import trinuc_fit
from nefertiti.functions.parse_mmcif import atomic_dtype
from crocodile.nuc import map_resname


def err(msg):
    print(msg, file=sys.stderr)
    exit(1)


NCHUNKS = 100
chunk = int(sys.argv[1])

rna_struc_index, rna_strucs_data = Buffer.load("allpdb-rna-attract").deserialize(
    "mixed"
)

rna_codes = []
rna_strucs = []
for code in rna_struc_index:
    ###if not code.startswith("1b7f") and not code.startswith("1cvj"): continue ###
    start, length = rna_struc_index[code]
    rna_strucs.append(rna_strucs_data[start : start + length])
    rna_codes.append(code)

bases = ("A", "C", "G", "U")
trinuc_sequences = ["".join(s) for s in itertools.product(bases, repeat=3)]

template_pdbs = {}
for seq in trinuc_sequences:
    filename = f"templates/{seq}-template-ppdb.npy"
    template = np.load(filename)
    if template.dtype != atomic_dtype:
        err(f"Template '{filename}' does not contain a parsed PDB")
    template_pdbs[seq] = template


conformers = {}
for seq in trinuc_sequences:
    filename = (
        f"/home/sjoerd/data/work/ProtNAff/database/trilib/{seq}-lib-conformer.npy"
    )
    conformer = np.load(filename)
    if conformer.dtype not in (np.float32, np.float64):
        err(f"Conformer file '{filename}' does not contain an array of numbers")
    if conformer.ndim != 3:
        err(f"Conformer file '{filename}' does not contain a 3D coordinate array")
    if conformer.shape[1] != len(template_pdbs[seq]):
        err(
            f"Sequence {seq}: conformer '{filename}' doesn't have the same number of atoms as the template"
        )
    conformers[seq] = conformer.astype(float)

rotaconformer_indices = {}
for seq in trinuc_sequences:
    seq2 = seq.replace("U", "C").replace("G", "A")
    if seq == seq2:
        filename = f"/home/sjoerd/data/work/crocodile/make-rotaconformers/results/{seq}-lib-rotaconformer.json"
        rotaconformer_index = Buffer.load(filename).deserialize("plain")
        if not isinstance(rotaconformer_index, list):
            err(f"Sequence {seq}: '{filename}' is not a list of filenames/checksums")
        if len(rotaconformer_index) != len(conformers[seq]):
            err(
                f"Sequence {seq}: There are {len(conformers[seq])} conformers but {len(rotaconformer_index)} rotaconformers"
            )
        for fnr, f in list(enumerate(rotaconformer_index)):
            rotaconformer_index[fnr] = os.path.join(
                # "/home/sjoerd/mbi-frontend/data3/sdevries/seamless/buffers", f
                "/data/sjoerd/rotaconformer-buffers",
                f,
            )
        rotaconformer_indices[seq] = rotaconformer_index
for seq in trinuc_sequences:
    seq2 = seq.replace("U", "C").replace("G", "A")
    rotaconformer_indices[seq] = rotaconformer_indices[seq2]

trinuc_rotaconformer_library = {}
for seq in rotaconformer_indices:
    for n, filename in enumerate(rotaconformer_indices[seq]):
        trinuc_rotaconformer_library[seq, n] = filename

chunk_indices = [n for n in range(len(rna_codes)) if (n % NCHUNKS) == (chunk - 1)]
print(len(chunk_indices))
rna_codes = [rna_codes[n] for n in chunk_indices]
rna_strucs = [rna_strucs[n] for n in chunk_indices]

mask = []
for n, struc in enumerate(rna_strucs):
    code = rna_codes[n]
    resnames = np.unique(struc["resname"])
    ok = True
    try:
        for resname in resnames:
            map_resname(resname, rna=True)
    except KeyError:
        ok = False
    mask.append(ok)

rna_codes = [code for n, code in enumerate(rna_codes) if mask[n]]
rna_strucs = [struc for n, struc in enumerate(rna_strucs) if mask[n]]

result = trinuc_fit(
    rna_strucs,
    rna_codes,
    template_pdbs=template_pdbs,
    trinuc_conformer_library=conformers,
    trinuc_rotaconformer_library=rotaconformer_indices,
    rmsd_margin1=0.5,
    rmsd_margin2=0.5,
    conformer_rmsd_min=0.1,
    conformer_rmsd_max=1.0,
)

# np.save(f"allpdb-overlap-rmsd-chunk-{chunk}.npy", result)
print(result[0])
buf = Buffer(result, celltype="mixed")
buf.save(f"allpdb-trinuc-fit-chunk-{chunk}")

import numpy as np

bases = ["A", "C", "G", "U"]
motifs = []
for b1 in bases:
    for b2 in bases:
        motifs.append(b1 + b2)

libsize = {}
for submotif in ("AA", "AC", "CA", "CC"):
    libf = f"library/library/dinuc-{submotif}-0.5.npy"
    libsize[submotif] = len(np.load(libf))

for motif in motifs:
    submotif = motif.replace("G", "A").replace("U", "C")
    libsize[motif] = libsize[submotif]

fitting = {}
for motif in motifs:
    sub_fitting = np.loadtxt(f"allpdb-rna-fit-dinuc-{motif}.txt")
    assert sub_fitting.shape[1] == 4
    all_pos = sub_fitting[:, 0].astype(np.uint)
    all_conf = sub_fitting[:, 1].astype(np.uint)
    all_rmsd = sub_fitting[:, 2].astype(float)
    is_replacement = sub_fitting[:, 3].astype(bool)

    for pos, conf, rmsd in zip(all_pos, all_conf, all_rmsd):
        if conf == 0:
            continue
        if conf > libsize[motif] and rmsd < 0.49:
            # TODO: continue not here, but if RMSD is huuuge
            # a bit of an artifact...
            # print(motif, pos[n], rmsd[n])
            continue
        assert pos not in fitting
        fitting[pos] = motif, conf, rmsd

min_pos = int(min(fitting.keys()))
max_pos = int(max(fitting.keys()))

pos = min_pos
seq = "XXXX" + fitting[pos][0][0]
chain = (-1, -1, -1, -1)
chain_ok = (True, True, True, True)

penta_redundant = {}
penta = {}


def add_penta(seq, chain):
    if seq not in penta:
        penta[seq] = set()
        penta_redundant[seq] = 0
    penta[seq].add(tuple(int(c) for c in chain))
    penta_redundant[seq] += 1


while 1:

    while pos in fitting:
        cur = fitting[pos]
        dinuc = cur[0]
        assert seq[-1] == dinuc[0]
        seq = seq[1:] + dinuc[1]
        conf = cur[1]
        rmsd = cur[2]
        is_ok = conf <= libsize[dinuc] and rmsd < 1
        chain = chain[1:] + (conf,)
        chain_ok = chain_ok[1:] + (is_ok,)
        if all(chain_ok):
            add_penta(seq, chain)
        # print(conf, rmsd, seq, all(chain_ok))  ###
        pos += 1

    for n in range(3):
        seq = seq[1:] + "X"
        chain = chain[1:] + (-1,)
        chain_ok = chain_ok[1:] + (1,)
        if all(chain_ok):
            add_penta(seq, chain)

        # print(-1, 0, seq, all(chain_ok))  ###

    if pos == max_pos + 1:
        break

    while pos not in fitting:
        pos += 1
    seq = "XXXX" + fitting[pos][0][0]
    chain = (-1, -1, -1, -1)
    chain_ok = (True, True, True, True)


keys_nx = {}
for k in penta:
    nx = k.count("X")
    keys_nx[k] = nx
for nx in range(4):
    keys = [k for k, v in keys_nx.items() if v == nx]
    nchain = sum([penta_redundant[k] for k in keys])
    n_nonre = sum([len(penta[k]) for k in keys])
    print(
        f"Number of X: {nx}, nchain {nchain}, nonredundant {n_nonre}, {n_nonre/nchain * 100:.2f} %"
    )

nchain = sum(penta_redundant.values())
n_nonre = sum([len(v) for v in penta.values()])
print()
print(f"Total, nchain {nchain}, nonredundant {n_nonre}, {n_nonre/nchain * 100:.2f} %")

import json

for k in list(penta.keys()):
    penta[k] = sorted(list(penta[k]))
with open("penta.json", "w") as f:
    json.dump(penta, f, sort_keys=True, indent=2)

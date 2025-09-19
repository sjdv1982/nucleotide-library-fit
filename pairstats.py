import json
from matplotlib import pyplot as plt
import numpy as np

with open("penta.json") as f:
    penta = json.load(f)

motifs = ("AA", "AC", "CA", "CC")
pair_motifs = ("AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC")

counts = {motif: {} for motif in motifs}

for seq, chains in penta.items():
    for pos in range(4):
        dinuc = seq[pos : pos + 2]
        if dinuc[0] == "X" or dinuc[1] == "X":
            continue
        motif = dinuc.replace("G", "A").replace("U", "C")
        mcounts = counts[motif]
        for chain in chains:
            conf = chain[pos]
            assert conf > 0, conf
            if conf not in mcounts:
                mcounts[conf] = 0
            mcounts[conf] += 1

pmotifs = ("AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC")
pcounts = {pmotif: {} for pmotif in pmotifs}
psums = {pmotif: 0 for pmotif in pmotifs}

for seq, chains in penta.items():
    for pos in range(3):
        trinuc = seq[pos : pos + 3]
        if trinuc[0] == "X" or trinuc[1] == "X" or trinuc[2] == "X":
            continue
        pmotif = trinuc.replace("G", "A").replace("U", "C")
        psums[pmotif] += len(chains)
        for chain in chains:
            subchain = (chain[pos] - 1, chain[pos + 1] - 1)
            if subchain not in pcounts[pmotif]:
                pcounts[pmotif][subchain] = 0
            pcounts[pmotif][subchain] += 1


bins = (
    [-9999, 0.2, 0.25]
    + list(np.arange(0.3, 0.7 + 0.01, 0.01))
    + [0.75, 0.80, 0.85, 9999]
)


def hist(r, *, weights=None):
    result = np.histogram(r, bins=bins, weights=weights)
    result = result[0].astype(float)
    return result / result.sum()


def pr(h):
    assert len(h) == len(bins) - 1
    for bin, hh in zip(bins[1:], h):
        print(f"{bin:.2f} {hh*100:.2f}")
    print()


def plot(h, plotfile):
    plt.plot([0] + bins[1:-1], h)
    plt.savefig(plotfile)
    plt.clf()


"""
for motif in motifs:
    print(motif)
    print(sum(counts[motif].values()))
    print(sorted(np.unique(list(counts[motif].values()), return_counts=True)[1])[-20:])
    print()
print(psums)
print()
"""

all_h0 = {}
all_h1 = {}
all_h2 = {}
all_proba_weights = {}

for pair_motif in pair_motifs:
    m1 = pair_motif[:2]
    m2 = pair_motif[-2:]

    crmsd = np.load(f"crmsd_matrix_{pair_motif}.npy")
    curr_counts1 = np.zeros(max(counts[m1].keys()))
    for k, v in counts[m1].items():
        assert k > 0
        curr_counts1[k - 1] = v
    curr_counts2 = np.zeros(max(counts[m2].keys()))
    for k, v in counts[m2].items():
        assert k > 0
        curr_counts2[k - 1] = v

    curr_pcountweights = np.zeros_like(crmsd)
    for (c1, c2), v in pcounts[m1 + m2[1]].items():
        curr_pcountweights[c1, c2] = v

    h0 = hist(crmsd)
    h0 /= h0.sum()
    plot(h0, f"pairstats-crmsd-{pair_motif}-raw.png")

    h1 = hist(crmsd, weights=curr_pcountweights)
    h1 /= h1.sum()
    plot(h1, f"pairstats-crmsd-{pair_motif}-obs.png")
    all_h1[pair_motif] = h1

    curr_proba_weights = curr_counts1[:, None] * curr_counts2[None, :]
    h2 = hist(crmsd, weights=curr_proba_weights)
    h2 /= h2.sum()
    plot(h2, f"pairstats-crmsd-{pair_motif}-expect.png")
    all_h2[pair_motif] = h2

    all_proba_weights[pair_motif] = curr_proba_weights


# h3 is the crmsd propensity: the relative probability factor given by a certain cRMSD, correcting for the proba pair product
h1 = np.zeros(len(bins) - 1)
h2 = np.zeros(len(bins) - 1)
for pair_motif in pair_motifs:
    h1 += all_h1[pair_motif]
    h2 += all_h2[pair_motif]
h3 = h1 / h2
pr(h3)
plot(h3, f"pairstats-crmsd-propensity.png")

for pair_motif in pair_motifs:
    m1 = pair_motif[:2]
    m2 = pair_motif[-2:]

    crmsd = np.load(f"crmsd_matrix_{pair_motif}.npy")

    crmsd_disc = np.digitize(crmsd, bins)
    crmsd_propensity = h3[crmsd_disc - 1]

    curr_proba_weights = all_proba_weights[pair_motif]
    est_proba = curr_proba_weights * crmsd_propensity
    est_proba /= est_proba.sum()

    est_proba_top = est_proba.argsort(axis=None)[::-1]
    est_proba_top_conf1, est_proba_top_conf2 = np.unravel_index(
        est_proba_top, est_proba.shape
    )

    nobs = (
        psums[m1 + m2[1]] / 3
    )  # every trinuc pair is replicated 4 times: XXAAA, XAAAX, XXAAA

    tot_miss_proba = 0
    tot_missed = 0
    with open(f"pairstats-{pair_motif}-antipairs.txt", "w") as f:
        print(f"#conf_{m1} #conf_{m2} #expected_count #miss_probability", file=f)
        for conf1, conf2 in zip(est_proba_top_conf1, est_proba_top_conf2):
            curr_est_proba = est_proba[conf1, conf2]
            expect = nobs * curr_est_proba

            # if expect < 5:  # at least 5 counts expected
            #    break

            miss_proba = (1 - curr_est_proba) ** nobs

            # TEST
            # what if we assumed we had 6k independent observations instead of the 26k we really have...
            ### miss_proba = (1 - curr_est_proba) ** 6000  ###
            # / TEST

            if miss_proba > 0.1:
                break
            elif miss_proba > 0.01:
                continue

            tot_miss_proba += miss_proba
            observed = int(curr_pcountweights[conf1, conf2] / 3)
            if observed == 0:
                print(f"{conf1 + 1} {conf2 + 1} {expect:.1f} {miss_proba:.4f}", file=f)
                tot_missed += 1

    print(pair_motif)
    print(
        "Expected to be missed (among those with <1 % chance to miss):",
        tot_miss_proba,
    )
    print(
        "Observed to be missed (among those with <1 % chance to miss):",
        tot_missed,
    )
    print()

MIN_NCLASSES = 5  # at least 5 word classes

import numpy as np

frags = np.load("allpdb-rna-fit-simplified.npy")
print(np.unique(frags["sequence"], return_counts=True))
seq = np.array([s.replace("G", "A").replace("U", "C") for s in frags["sequence"]])
print(np.unique(seq, return_counts=True))
pairs = np.load("allpdb-rna-fit-pairs.npy")
weights = 0.5 * np.bincount(
    pairs["index"] - 1, minlength=len(frags)
) + 0.5 * np.bincount(pairs["index"], minlength=len(frags))

frag_classes = np.zeros(len(frags), int)
frag_classes[:] = -1
nclasses = {}
conf_classes = {}
lib_class_thresholds = {}
for lib in ("AA", "AC", "CA", "CC"):
    mask = seq == lib
    curr_frags = frags[mask]
    curr_weights = weights[mask]
    curr_frag_classes = np.zeros(len(curr_frags))
    nconf = curr_frags["conformer"].max() + 1
    conf_counts = np.zeros(nconf)
    np.add.at(conf_counts, curr_frags["conformer"], curr_weights)
    conf_classes[lib] = np.zeros(nconf)

    sorting = np.argsort(conf_counts)[::-1]
    sortcounts = conf_counts[sorting]
    cum_sortcounts = np.cumsum(sortcounts) / curr_weights.sum() * 100

    # inefficient, but who cares...
    curr_class = 0
    threshold = 20
    positions = [0]
    for pos in range(nconf):
        if sortcounts[pos] < 5:
            positions.append(pos)
            break
        v = cum_sortcounts[pos]
        if v < threshold:
            continue
        curr_class += 1
        positions.append(pos + 1)
        if curr_class == MIN_NCLASSES - 1:
            break
        else:
            threshold = v + (100 - v) / (MIN_NCLASSES - curr_class)
    for pos in range(nconf):
        if sortcounts[pos] < 5:
            positions[-1] = pos
            break
    positions.append(nconf)
    lib_class_thresholds[lib] = sortcounts[np.array(positions) - 1][::-1]
    lib_class_thresholds[lib][-1] = np.inf

    nclasses[lib] = len(positions) - 1
    for n in range(nclasses[lib]):
        tmin = lib_class_thresholds[lib][n]
        tmax = lib_class_thresholds[lib][n + 1]
        tmask = (conf_counts >= tmin) & (conf_counts < tmax)
        conf_classes[lib][tmask] = n

    curr_frag_classes[:] = conf_classes[lib][curr_frags["conformer"]]
    frag_classes[mask] = curr_frag_classes

for lib in ("AA", "AC", "CA", "CC"):
    mask = seq == lib
    curr_frags = frags[mask]
    curr_frag_classes = frag_classes[mask]
    curr_frags = frags[mask]
    nconf = curr_frags["conformer"].max() + 1

    for n in range(nclasses[lib]):
        class_nconfs = (conf_classes[lib] == n).sum()
        class_nfrags = (curr_frag_classes == n).sum()
        print(
            f"Lib {lib}, class {n}, nconfs {class_nconfs} ({class_nconfs/nconf*100:.1f} %), nfrags {class_nfrags} ({class_nfrags/len(curr_frags)*100:.2f}) %, {lib_class_thresholds[lib][n]:.1f} <= counts < {lib_class_thresholds[lib][n+1]:.1f}"
        )
    print()

max_nclasses = max(nclasses.values())

masknuc1 = {}
masknuc2 = {}
classfreq_nuc1 = {}
classfreq_nuc2 = {}

seqnuc1 = seq[pairs["index"] - 1]
seqnuc2 = seq[pairs["index"]]
for lib in ("AA", "AC", "CA", "CC"):
    masknuc1[lib] = seqnuc1 == lib
    masknuc2[lib] = seqnuc2 == lib
    s1 = masknuc1[lib].sum()
    s2 = masknuc2[lib].sum()
    for class_ in range(max_nclasses):
        class1mask = frag_classes[pairs["index"] - 1] == class_
        class2mask = frag_classes[pairs["index"]] == class_
        classfreq_nuc1[lib, class_] = (masknuc1[lib] & class1mask).sum() / s1
        classfreq_nuc2[lib, class_] = (masknuc2[lib] & class2mask).sum() / s2

tot = 0
tot_exp = 0
propensity = {}
for class1 in range(max_nclasses):
    class1mask = frag_classes[pairs["index"] - 1] == class1
    for class2 in range(max_nclasses):
        class2mask = frag_classes[pairs["index"]] == class2
        exp_sum = 0
        for pairlib in ("AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"):
            lib1 = pairlib[:2]
            lib2 = pairlib[-2:]
            nucmask = masknuc1[lib1] & masknuc2[lib2]
            s = nucmask.sum()
            mask = class1mask & class2mask & nucmask
            pairlib_expected = (
                classfreq_nuc1[lib1, class1] * classfreq_nuc2[lib2, class2] * s
            )
            exp_sum += pairlib_expected

        obs = (class1mask & class2mask).sum()
        tot += obs
        tot_exp += exp_sum
        pairlib_observed = mask.sum()
        propensity[class1, class2] = obs / exp_sum
        print(
            f"Word {class1}{class2}, observed {obs} expected {exp_sum:.1f} propensity {propensity[class1,class2]:.3f}"
        )

print()
print("CHECK (3x same number)", tot, int(np.round(tot_exp)), len(pairs))
print()

print(f"Length 3 analysis: {max_nclasses**3} words")


p1 = pairs[:-1]
p2 = pairs[1:]
pmask = (p2["index"] - p1["index"]) == 1
inds = p1[pmask]["index"] - 1
print(pmask.sum())

exp_raw = {}
exp_prop = {}
obs = {}
for class1 in range(max_nclasses):
    class1mask = frag_classes[inds] == class1
    freq1 = class1mask.mean()
    for class2 in range(max_nclasses):
        class2mask = frag_classes[inds + 1] == class2
        freq2 = class2mask.mean()
        for class3 in range(max_nclasses):
            class3mask = frag_classes[inds + 2] == class3
            freq3 = class3mask.mean()
            exp = freq1 * freq2 * freq3 * len(inds)
            exp_raw[class1, class2, class3] = exp
            exp_prop[class1, class2, class3] = (
                exp * propensity[class1, class2] * propensity[class2, class3]
            )
            curr_obs = (class1mask & class2mask & class3mask).sum()
            obs[class1, class2, class3] = curr_obs
print(len(inds), sum(exp_raw.values()))
factor = sum(exp_prop.values()) / len(inds)

count = 0
outliers = {}
for class1 in range(max_nclasses):
    for class2 in range(max_nclasses):
        for class3 in range(max_nclasses):
            exp = exp_prop[class1, class2, class3] / factor
            ob = obs[class1, class2, class3]
            delta = np.abs(exp - ob)
            if delta > 20 and (ob / exp > 1.3 or exp / ob > 1.3):
                print(
                    count,
                    f"Word {class1}{class2}{class3} obs {ob} expected {exp:.0f} propensity {ob/exp:.2f}",
                )
                outliers[class1, class2, class3] = ob / exp
                count += 1

print()

import numpy as np

class_vector = np.zeros(max_nclasses)
print("class vector, averaged over the libraries and pair positions")
for n in range(max_nclasses):
    stats = [classfreq_nuc1[k] for k in classfreq_nuc1.keys() if k[1] == n] + [
        classfreq_nuc2[k] for k in classfreq_nuc2.keys() if k[1] == n
    ]
    stats = np.array(stats)
    print(f"class {n}, raw value: {stats.mean():.3f}, stdev: {stats.std():.3f}")
    class_vector[n] = stats.mean()
print("Raw sum:", class_vector.sum())
class_vector /= class_vector.sum()
np.save("class_vector.npy", class_vector)

print()
print("Store 2-word propensity matrix")

freq_2d = np.zeros((max_nclasses, max_nclasses))
propensity_2d = np.zeros((max_nclasses, max_nclasses))
for class1 in range(max_nclasses):
    class1mask = frag_classes[pairs["index"] - 1] == class1
    for class2 in range(max_nclasses):
        class2mask = frag_classes[pairs["index"]] == class2
        freq = (class1mask & class2mask).mean()
        freq_2d[class1, class2] = freq
        propensity_2d[class1, class2] = freq / (
            class_vector[class1] * class_vector[class2]
        )

assert abs(freq_2d.sum() - 1) < 0.01
np.save("propensity_2d.npy", propensity_2d)
print()

print("Store 3-word propensity matrix")

propensity_3d = np.zeros((max_nclasses, max_nclasses, max_nclasses))
for class1 in range(max_nclasses):
    for class2 in range(max_nclasses):
        for class3 in range(max_nclasses):
            prop = outliers.get((class1, class2, class3), 1)
            propensity_3d[class1, class2, class3] = prop
np.save("propensity_3d.npy", propensity_3d)

print()

print("Write out fragment classes")
for lib in ("AA", "AC", "CA", "CC"):
    np.save(f"conformer-classes-{lib}.npy", conf_classes[lib])


# Filter for contacts

frag_contacts = np.load("../nucleotide-interaction/allpdb-rna-fit-count-contacts.npy")
frag_contacts_mask = frag_contacts > 0
frag_reindex = np.cumsum(frag_contacts_mask)
frags = frags[frag_contacts_mask]
frag_classes = frag_classes[frag_contacts_mask]
seq = seq[frag_contacts_mask]

contact_p1 = frag_contacts[pairs["index"] - 1] > 0
contact_p2 = frag_contacts[pairs["index"]] > 0
pair_contact_mask = contact_p1 & contact_p2

pairs = pairs[pair_contact_mask]
new_index = frag_reindex[pairs["index"] - 1] - 1
new_index2 = frag_reindex[pairs["index"]] - 1
assert np.all((new_index2 - new_index) == 1)

pairs["index"] = new_index2
print()

# /filter

print("Repeat analysis after filtering for contacting fragments")
print()
for lib in ("AA", "AC", "CA", "CC"):
    mask = seq == lib
    curr_frags = frags[mask]
    curr_frag_classes = frag_classes[mask]
    for n in range(nclasses[lib]):
        class_nconfs = (conf_classes[lib] == n).sum()
        class_nfrags = (curr_frag_classes == n).sum()
        print(
            f"Lib {lib}, class {n}, nconfs {class_nconfs} ({class_nconfs/nconf*100:.1f} %), nfrags {class_nfrags} ({class_nfrags/len(curr_frags)*100:.2f}) %"
        )
    print()


### /repeat

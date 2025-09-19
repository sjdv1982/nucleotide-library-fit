"""
Experiment to see how AA dinucleotides participate in trinucleotides. 
Only consider the primary libraries.
"""

import numpy as np
from tqdm import tqdm
from clusterlib import read_clustering
from nefertiti.functions.superimpose import superimpose_array

precision = 0.5
sizeA = 22


coor = np.load(f"library/dinuc-AA-{precision}.npy")
clustering = read_clustering(f"lib-dinuc-AA-{precision}.clust")
coor_indices = np.loadtxt(
    f"build-library-dinuc-AA-{precision}-primary-indices.txt"
).astype(int)

# probability vector
ind_to_cluster = {}
for clusnr, clus in enumerate(clustering):
    for ind in clus:
        ind_to_cluster[ind] = clusnr
assert len(coor_indices) == len(coor)
coor_clusters = [ind_to_cluster[ind] for ind in coor_indices]
cluster_lengths = np.array([len(clustering[c]) for c in coor_clusters])
prob_vector = cluster_lengths / cluster_lengths.sum()

trinuc_coor = {}
motifs = ("AAA", "AAC", "CAA")
for motif in motifs:
    trinuc_coor[motif] = np.load(f"library/trinuc-{motif}-{precision}.npy")

trinuc_part = np.zeros(len(coor), int)

pairs = np.full((len(trinuc_coor["AAA"]), 2), -1).astype(int)

for motif in motifs:
    if motif[:2] != "AA":
        continue
    coor3 = trinuc_coor[motif][:, : 2 * sizeA]
    for strucnr, struc in enumerate(tqdm(coor3)):
        _, rmsd = superimpose_array(coor, struc)
        if rmsd.min() > precision:
            print("non-existent", motif, strucnr, "pre")
            continue
        best = rmsd.argmin()
        trinuc_part[best] += 1
        if motif == "AAA":
            pairs[strucnr, 0] = best

for motif in motifs:
    if motif[-2:] != "AA":
        continue
    coor3 = trinuc_coor[motif][:, -2 * sizeA :]
    for strucnr, struc in enumerate(tqdm(coor3)):
        _, rmsd = superimpose_array(coor, struc)
        if rmsd.min() > precision:
            print("non-existent", motif, strucnr, "post")
            continue
        best = rmsd.argmin()
        trinuc_part[best] += 1
        if motif == "AAA":
            pairs[strucnr, 1] = best

# overlap RMSD
overlap_zone = coor.shape[1] - sizeA
pre = coor[:, -overlap_zone:]
post = coor[:, :overlap_zone]
rmsd = np.empty((len(coor), len(coor)))
rotmat = np.empty((len(coor), len(coor), 3, 3))
for n in tqdm(list(range(len(coor)))):
    m, r = superimpose_array(pre, post[n])
    rmsd[:, n] = r
    rotmat[:, n] = m


rs = []
for p1, p2 in pairs:
    if p1 != -1 and p2 != -1:
        rs.append(rmsd[p1, p2])
rs = np.array(rs)
print(np.histogram(rs))
print((rs < precision).mean())

"""
Answer: out of 2276 (non-singleton) dinucleotides, 529 (almost a quarter) are never part of a trinucleotide.
Of those 529, 277 have cluster length 1 (must be multi-PDB unique confs), and only 27 have cluster length >=4

97.6 % of the AAA library consists of 2 AA conformers that fit better than 1A. 
89.1 % of the AAA library consists of 2 AA conformers that fit better than 1A and whose overlap RMSD is better than 1A. 
        (96.0 % for overlap RMSD under 1.2 A; worst overlap RMSD 1.52A)
These 89.1 % are 2028 conformers.
There are 960142 AA-AA pairings with an overlap RMSD under 1A
This means that only 0.21 % of the possible AA-AA pairings is really observed as a trinucleotide

"""

ccoor = coor - pre.mean(axis=1)[:, None, :]

cpost = post - post.mean(axis=1)[:, None, :]
ccoor2 = coor - post.mean(axis=1)[:, None, :]

coor_sup = np.zeros((len(coor), len(coor)) + coor.shape[1:], dtype=np.float32)
for nn in tqdm(list(range(len(coor)))):
    x2 = np.einsum("ijk,ikl->ijl", ccoor, rotmat[:, nn])
    coor_sup[:, nn] = x2


def rms(c):
    return np.sqrt((c * c).sum() / len(c))


clashing = np.zeros((len(coor), len(coor)), bool)

for n in tqdm(list(range(len(coor)))):
    for nn in range(len(coor)):
        assert (
            np.abs(
                rms(coor_sup[n, nn][-overlap_zone:] - ccoor2[nn][:overlap_zone])
                - rmsd[n, nn]
            )
            < 0.001
        )
        res1 = coor_sup[n, nn][:sizeA]
        res2 = ccoor2[nn][-sizeA:]
        d = res1[:, None, :] - res2[None, :, :]
        dis = np.sqrt((d * d).sum(axis=2))
        mindis = dis.min()
        if mindis < 2:
            clashing[n, nn] = 1

rs2 = []
for p1, p2 in pairs:
    if p1 != -1 and p2 != -1 and not clashing[p1, p2]:
        rs2.append(rmsd[p1, p2])
rs2 = np.array(rs2)
print(np.histogram(rs2))
print((rs2 < precision).mean())

# Of the 2028, 2004 have no clashes
# This is 0.22 % of the non-clashing <1A transitions
# In other words, 7.70 % of the <1A transitions is clashing
#   but 1.18 % of the trinuc-observed <1A AA-AA superpositions is also clashing

#######################
# 0.5 A precision
#######################

#99.1 % of the AAA library consists of 2 AA conformers that fit better than 0.5A. 
#90.2 % of the AAA library consists of 2 AA conformers that fit better than 0.5A and whose overlap RMSD is better than 0.5A. 
#        (98.4 % for overlap RMSD under 0.7 A; worst overlap RMSD 0.865A)
#
# Long version: there are 5119 primary trinucleotides
# Of those, 5073 (99.1 %) consist of a pair of two primary dinucleotides that fit *to the trinucleotide* at 0.5A
#       (aside: all but 5 (5114) consist of a pair of two primary dinucleotides that fit at 1A)
#   90.2 % of the AAA library consists of 2 AA conformers that fit better than 0.5 A *to each other* (overlap RMSD)
#       ( i.e. <0.5A transitions). 
#   (96.7 % fit better than 0.6 A, and 98.4 % better than 0.7 A (max possible is 99.1%); worst overlap RMSD 0.865A)
# For pairs of primary AA dinucleotides (i.e. AA-AA superpositions):
#  - among any pair (not just the <0.5A transitions), clashing is uncommon (4.3 %), 
#  - among the <0.5 A transitions it is rare
#       (14729 out of 2087187, or 0.71 %) (compare to 7.70 % for 1A precision)
#  - among the trinuc-observed AA-AA superpositions, it is very rare
#       (10 out of 5073, or 0.22 % )
#  - among the trinuc-observed <0.5A AA-AA superpositions, it is also very rare
#       (9/4629, or 0.22 %)  (compare to 1.18 % for 1A precision).
#
# Among all (2072458) <0.5A non-clashing AA-AA superpositions , 
# the 4629 trinuc-observed ones are but a small fraction
# (0.22 %, the exact same percentage as for 1A precision)
# 
# For <0.7A transitions: 2.7x (5621157) as many possible transition as for <0.5A
#   1.15 % of all transitions is clashing
# Among the non-clashing ones,
#    the 5040 trinuc-observed ones are therefore an even smaller fraction (0.091 %)

# only 3356 out of 6073 primary dinucleotides are part of any primary trinucleotide at all
print(len(np.unique(pairs))-1, len(coor))

pairs_mask = (pairs[:, 0] > -1) & (pairs[:, 1] > -1)
pairs2 = pairs[pairs_mask]
log_pre_counts = {k:np.log(v/len(pairs2)) for k, v in zip(*np.unique(pairs2[:, 0], return_counts=True))}
log_post_counts = {k:np.log(v/len(pairs2)) for k, v in zip(*np.unique(pairs2[:, 1], return_counts=True))}
obs_pairs = set([(p[0],p[1]) for p in pairs2])

lp = len(obs_pairs)
obs_probs = []
obs_real = []
obs_inds = []
norm = 0
for k1, v1 in tqdm(log_pre_counts.items()):
    for k2, v2 in log_post_counts.items():
        p = np.exp(v1+v2)
        if rmsd[k1, k2] > 0.7:
            p = 0
        elif rmsd[k1, k2] > 0.5:
            p /= 18.7
        norm += p

for k1, v1 in tqdm(log_pre_counts.items()):
    for k2, v2 in log_post_counts.items():
        p = np.exp(v1+v2)
        if rmsd[k1, k2] > 0.7:
            p = 0
        elif rmsd[k1, k2] > 0.5:
            p /= 18.7
        p /= norm

        non_obs = (1-p)**lp

        is_obs = (k1, k2) in obs_pairs
        obs_probs.append(1-non_obs)
        obs_real.append(is_obs)
        obs_inds.append((k1, k2))
obs_probs = np.array(obs_probs)        
obs_prob_sort = np.argsort(obs_probs)[::-1]
obs_probs = obs_probs[obs_prob_sort]
obs_real = np.array(obs_real)[obs_prob_sort]
obs_inds = np.array(obs_inds)[obs_prob_sort]
obs_real_count = np.cumsum(obs_real)

# Of the 10 combinations that are "most obvious" (with the highest combined expected frequency), only 6 are observed
# Obviousness is based on how many trinuc transition pairs are observed with that conformer, *not* the number of fragments in the cluster! 
print("#Conf1 #conf2 #is_real")
for o_inds, o_real in zip(obs_inds[:10], obs_real[:10]):
    print(o_inds, o_real)
    assert (tuple(o_inds) in obs_pairs) == o_real
print(f"Total: {obs_real[:10].sum()}")
print(f"Expected real pairs in the top 10 most obvious: {obs_probs[:10].sum():.1f}")
print()

# This discrepancy continues for the top 100...

print(f"Top 100 most obvious pairs: {obs_real_count[100]} are observed")
print(f"Expected real pairs in the top 100 most obvious: {obs_probs[:100].sum():.1f}")
print()

# ... and for the top 1000 
# (SEE BELOW: if overlap RMSD is not considered, obviousness is an excellent predictor beyond the top 100)

print(f"Rank 100-1000 most obvious pairs: {obs_real[100:1000].sum()} are observed")
print(f"Expected real pairs: {obs_probs[100:1000].sum():.1f}")
print()

# Taking the #N most obvious pairs (where N is the number of trinucleotides) yields only 10 % of the observed pairs
# (and only 10 % of the most obvious pairs are observed)   
print(f"#N (N=number of observed trinucleotides={lp}) most obvious pairs: {obs_real_count[lp]/lp*100:.1f} % are observed")
print(f"Expected real pairs in the top #N most obvious: {obs_probs[:lp].sum()/lp*100:.1f} %")
print()

# Even increasing the number of selected most obvious pairs five-fold, only increases the observed pairs two-fold
print(f"(5x #N) most obvious pairs: {obs_real_count[5*lp]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 5x #N most obvious: {obs_probs[:5*lp].sum()/lp*100:.1f} %")
print()

# You need 100k (~30 per conformer) to get half of the trinucleotides
print(f"100k most obvious pairs: {obs_real_count[100000]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 100k most obvious: {obs_probs[:100000].sum()/lp*100:.1f} %")
print()

# You need 1 million (~300 per conformer, half of <0.5A) to get 99 % of the trinucleotides
print(f"1 million most obvious pairs: {obs_real_count[1000000]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 1 million most obvious: {obs_probs[:1000000].sum()/lp*100:.1f} %")
print()


"""
Conclusion
A priori, there are 36 million pairings
Only 2 million (1:18) are allowed based on overlap geometry (<0.5A, 90 % of the trinuc library)
    5 million (1:7) are generously allowed based on overlap geometry (<0.5A, 98.4 % of the trinuc library)\
Only 4811 are observed (5073 trinuc, but some identical pairings), which is ~1:400 of 2 million
    99 % are in the 1 million most "obvious" pairs, i.e. dinuc that are common in trinuc
    even so, for the 100 most obvious pairings, only 60 are observed (expected: 94), the other 40 can be blacklisted.
    This "obviousness" is *extremely* conservative as it assumes that every trinuc is seen only once.
"""

print("OBVIOUSNESS ALONE, NOT CONSIDERING OVERLAP RMSD")
print()

lp = len(obs_pairs)
obs_probs = []
obs_real = []
obs_inds = []
norm = 0
for k1, v1 in tqdm(log_pre_counts.items()):
    for k2, v2 in log_post_counts.items():
        p = np.exp(v1+v2)
        norm += p

for k1, v1 in tqdm(log_pre_counts.items()):
    for k2, v2 in log_post_counts.items():
        p = np.exp(v1+v2)
        p /= norm

        non_obs = (1-p)**lp

        is_obs = (k1, k2) in obs_pairs
        obs_probs.append(1-non_obs)
        obs_real.append(is_obs)
        obs_inds.append((k1, k2))
obs_probs = np.array(obs_probs)        
obs_prob_sort = np.argsort(obs_probs)[::-1]
obs_probs = obs_probs[obs_prob_sort]
obs_real = np.array(obs_real)[obs_prob_sort]
obs_inds = np.array(obs_inds)[obs_prob_sort]
obs_real_count = np.cumsum(obs_real)

# Again, Of the 10 combinations that are "most obvious" (with the highest combined expected frequency), only 6 are observed
print("#Conf1 #conf2 #is_real")
for o_inds, o_real in zip(obs_inds[:10], obs_real[:10]):
    print(o_inds, o_real)
    assert (tuple(o_inds) in obs_pairs) == o_real
print(f"Total: {obs_real[:10].sum()}")
print(f"Expected real pairs in the top 10 most obvious: {obs_probs[:10].sum():.1f}")
print()

# Again, this discrepancy continues for the top 100...

print(f"Top 100 most obvious pairs: {obs_real_count[100]} are observed")
print(f"Expected real pairs in the top 100 most obvious: {obs_probs[:100].sum():.1f}")
print()

# ... but beyond the top 100, obviousness is now an excellent predictor

print(f"Rank 100-1000 most obvious pairs: {obs_real[100:1000].sum()} are observed")
print(f"Expected real pairs: {obs_probs[100:1000].sum():.1f}")
print()

# Taking the #N most obvious pairs (where N is the number of trinucleotides) yields only 10 % of the observed pairs
# (and only 10 % of the most obvious pairs are observed)   
print(f"#N (N=number of observed trinucleotides={lp}) most obvious pairs: {obs_real_count[lp]/lp*100:.1f} % are observed")
print(f"Expected real pairs in the top #N most obvious: {obs_probs[:lp].sum()/lp*100:.1f} %")
print()

# Even increasing the number of selected most obvious pairs five-fold, only increases the observed pairs two-fold
print(f"(5x #N) most obvious pairs: {obs_real_count[5*lp]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 5x #N most obvious: {obs_probs[:5*lp].sum()/lp*100:.1f} %")
print()

# You need 100k (~30 per conformer) to get half of the trinucleotides
print(f"100k most obvious pairs: {obs_real_count[100000]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 100k most obvious: {obs_probs[:100000].sum()/lp*100:.1f} %")
print()

# You need 1 million (~300 per conformer, half of <0.5A) to get 99 % of the trinucleotides
print(f"1 million most obvious pairs: {obs_real_count[1000000]/lp*100:.1f} % of observed pairs are accounted for")
print(f"Expected real pairs in the top 1 million most obvious: {obs_probs[:1000000].sum()/lp*100:.1f} %")
print()



import itertools
import numpy as np
from seamless import Buffer
from tqdm import tqdm

sequences = ["".join(seq) for seq in itertools.product(('A','C'), repeat=4)]
comp_mat = {}
for seq in sequences:
    ###comp_mat[seq] = np.load(f"/home/sjoerd/data/work/ProtNAff/database/trilib/{seq}-compatibility-rmsd-full.npy")
    comp_mat[seq] = np.load(f"/home/sjoerd/data/work/ProtNAff/database/trilib/{seq}-compatibility-rmsd.npy")


comp_rmsds_good = []
comp_rmsds_bad = []
nsingleton = 0
ndoublesingleton = 0
for chunk in tqdm(list(range(100))):
    d = Buffer.load(f"allpdb-trinuc-fit-chunk-{chunk+1}").deserialize("mixed")
    d_index, d_data = d
    inds = [p[0]-1 for p in d_index.values() if p[0] > 0 and p[0] < len(d_data)-1]
    first = np.diff(d_data["first_resid"],prepend=-1).astype(bool)
    starts = [p[0] for p in d_index.values() if p[0] < len(d_data)]
    first[starts] = True
    data = d_data[first]

    for n in range(len(data)-1):
        r1, r2 = data[n:n+2]
        if r1["first_resid"] + 1 != r2["first_resid"]:
            continue
        assert r1["sequence"][1:] == r2["sequence"][:2]
        seq = (r1["sequence"] + r2["sequence"][2:3]).decode().replace("U", "C").replace("G", "A")
        comp_rmsd = comp_mat[seq][r1["conformer"], r2["conformer"]]
        if r1["rmsd"] < 1 and r2["rmsd"] < 1:
            comp_rmsds_good.append(comp_rmsd)
        else:
            if r1["rmsd"] > 1:
                nsingleton += 1
                if r2["rmsd"] > 1:
                    ndoublesingleton += 1
            comp_rmsds_bad.append(comp_rmsd)

comp_rmsds = np.array(comp_rmsds_good + comp_rmsds_bad)
print("All", len(comp_rmsds))
print(np.histogram(comp_rmsds))        
print()

comp_rmsds_good = np.array(comp_rmsds_good)
print("Good (RMSD < 1 for both of the pair)", len(comp_rmsds_good))
print(np.histogram(comp_rmsds_good)) 
print()

print("RMSD > 1 for the first one in the pair:", nsingleton)
print()
print("RMSD > 1 for the both in the pair:", ndoublesingleton)
print()


comp_rmsds_bad = np.array(comp_rmsds_bad)
print("Bad (RMSD >= 1 for at least one of the pair)", len(comp_rmsds_bad))
print(np.histogram(comp_rmsds_bad))        
print()
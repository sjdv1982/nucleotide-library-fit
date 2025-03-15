import numpy as np

frags = [3, 5, 6, 7, 8, 9]
for frag in frags:
    true = int(open(f"frag-{frag}-true.txt").read())
    crmsds = np.loadtxt(f"frag-{frag}-crmsd.txt")
    crmsd_mask = crmsds < 0.5
    print(
        frag, "cRMSD = {:.3f}".format(crmsds[true - 1]), f", {crmsd_mask.sum()} <0.5A"
    )
    for b in ("1b7f", "3sxl"):
        all_scores = []
        for dom in ("dom1", "dom2", "both"):
            if dom == "both":
                scores = np.stack(all_scores, axis=1).sum(axis=1)
            else:
                scores = np.loadtxt(f"frag-{frag}-{b}_{dom}.ene")[:, 1]
                all_scores.append(scores)
            assert scores.ndim == 1 and len(scores) == len(crmsds)
            ranks0 = scores.argsort()
            ranks = np.array([r for r in ranks0 if crmsd_mask[r]])
            rank = ranks.tolist().index(true - 1) + 1
            print(b, dom, rank, "/", crmsd_mask.sum())
        print()
    print()

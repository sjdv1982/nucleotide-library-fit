import numpy as np

fit2_dtype = np.dtype(
    [
        ("pdb", "U6"),
        ("fragment", np.uint16),
        ("resid", np.uint16),
        ("sequence", "U2"),
        ("conformer", np.int16),
        ("rmsd", np.float32),
        ("ermsd", np.float32),
        ("drmsd", np.float32),
    ],
    align=True,
)
data = np.load("allpdb-rna-fit.npy")
data2 = np.empty(len(data), fit2_dtype)
data2["pdb"] = data["pdb"]
data2["fragment"] = data["fragment"]
data2["resid"] = data["resid"]
data2["sequence"] = data["sequence"]
data2["conformer"] = data["conformer"]
data2["rmsd"] = data["conf_rmsd"]
data2["ermsd"] = data["e-conf_rmsd"]
data2["drmsd"] = data["drmsd"]
np.save("allpdb-rna-fit-simplified.npy", data2)

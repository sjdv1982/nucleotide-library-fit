import numpy as np


def dinuc_fit(
    strucs: list[np.ndarray],
    codes: list,
    *,
    template_pdbs: dict[str, np.ndarray],
    dinuc_conformer_library: dict[str, np.ndarray],
    rmsd_margin: float,
    rmsd_soft_max: float
):
    # TODO: include crocodile/nefertiti as Seamless modules
    from crocodile.dinuc.from_ppdb import from_ppdb
    from crocodile.trinuc.best_fit import best_fit

    stage1 = {}
    for strucnr, struc in enumerate(strucs):
        code = codes[strucnr]
        dinuc_co = from_ppdb(
            struc,
            template_pdbs=template_pdbs,
            dinuc_conformer_library=dinuc_conformer_library,
            rna=True,
            ignore_unknown=False,
            ignore_missing=False,
            ignore_reordered=True,
            rmsd_margin=rmsd_margin,
            rmsd_soft_max=rmsd_soft_max,
        )
        bf = best_fit(dinuc_co)
        print(
            strucnr + 1,
            code,
            len(dinuc_co),
            len(bf),
            ["{:.3f}".format(r) for r in bf["rmsd"][:10]],
        )
        stage1[code] = dinuc_co

    result_data = []
    result_index = {}
    offset = 0
    for code, dinuc_co in stage1.items():
        result_data.append(dinuc_co)
        result_index[code] = (offset, len(dinuc_co))
        offset += len(dinuc_co)
    result_data = np.concatenate(result_data)
    return result_index, result_data

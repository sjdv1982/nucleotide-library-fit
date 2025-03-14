import numpy as np


def trinuc_fit(
    strucs: list[np.ndarray],
    codes: list,
    *,
    template_pdbs: dict[str, np.ndarray],
    trinuc_conformer_library: dict[str, np.ndarray],
    rmsd_margin: float,
    rmsd_soft_max: float
):
    # TODO: include crocodile/nefertiti as Seamless modules
    from crocodile.trinuc.from_ppdb import from_ppdb
    from crocodile.trinuc.best_fit import best_fit

    stage1 = {}
    for strucnr, struc in enumerate(strucs):
        code = codes[strucnr]
        trinuc_co = from_ppdb(
            struc,
            template_pdbs=template_pdbs,
            trinuc_conformer_library=trinuc_conformer_library,
            rna=True,
            ignore_unknown=False,
            ignore_missing=False,
            ignore_reordered=True,
            rmsd_margin=rmsd_margin,
            rmsd_soft_max=rmsd_soft_max,
        )
        bf = best_fit(trinuc_co)
        print(
            strucnr + 1,
            code,
            len(trinuc_co),
            len(bf),
            ["{:.3f}".format(r) for r in bf["rmsd"][:10]],
        )
        stage1[code] = trinuc_co

    result_data = []
    result_index = {}
    offset = 0
    for code, trinuc_co in stage1.items():
        result_data.append(trinuc_co)
        result_index[code] = (offset, len(trinuc_co))
        offset += len(trinuc_co)
    result_data = np.concatenate(result_data)
    return result_index, result_data

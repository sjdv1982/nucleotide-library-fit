for b in 1b7f 3sxl; do
    for dom in 1 2; do
        $ATTRACTDIR/make-grid-omp ${b}_dom${dom}r.pdb $ATTRACTDIR/../attract.par 5 7 ${b}_dom${dom}r.grid
    done
done
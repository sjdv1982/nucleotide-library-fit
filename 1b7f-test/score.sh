fragnr=$1   # e.g. 3
rec=$2  #e.g. ${rec}
python3 ~/data/work/attract-jax/test3.py frag-$fragnr-poses.npy ${rec}r-coor.npy frag-$fragnr-lib.npy \
    --conformers frag-$fragnr-conf.npy \
    --atrec ${rec}r-atomtypes.npy \
    --atlig frag-$fragnr-atomtypes.npy \
    --grid ${rec}r.grid \
    --output frag-$fragnr-${rec}.ene.npy
python -c '''
import sys, numpy as np; a = np.load(sys.argv[1])
for i,aa in enumerate(a): 
    print(i+1, f"{aa:3f}")
''' frag-$fragnr-${rec}.ene.npy > frag-$fragnr-${rec}.ene
rm -f frag-$fragnr-${rec}.ene.npy
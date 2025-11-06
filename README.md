# Analyses performed in this repo

## allpdb-rna-fit-dinuc.py
Create an RNA "monster sequence" (with seg breaks) and fits all library dinucleotides on it.  
The extension library is appended after the primary one.  
The library is filtered for same-PDB-origin (using the replacement library).

Store as `allpdb-rna-fit-dinuc-{motif}.txt`  
Motifs are A/C/G/U-A/C/G/U.

Format: each line contains the following columns:
- monster sequence position (starting at 1)
- best fit library conformer (starting at 1)
- best fit RMSD
- is_replacement: if the fragment comes from the replacement library (because the primary conformer had the same PDB origin)


## crmsd_matrix.py
Calculates the compatibility RMSD between each compatible pair of primary dinucleotide libraries.  
The compatibility between AC and CA is stored as `crmsd_matrix_ACA.npy`.


## penta.py
For each unique pentanucleotide sequence, all conformer chains (conf1, conf2, conf3, conf4) are enumerated.  
They are filtered for RMSD<1A for every conf, and only primary confs are considered.  
Terminal chains are padded with X, so that every conf contributes to four chains and every conf pair contributes to three chains.  
Every conformer chain is stored only once. Non-redundancy is about 55 % of all pentanucleotide conformer chains.


## pairstats.py
Analyses the non-redundant conf chains computed by penta.py.  
For the sake of statistics, A/C/G/U => A/C (the conf indices point to A/C libraries).

Calculates, for every A/C dinuc sequence:
- the observed frequency of every conf (`freq-{dinuc}.txt`)

Calculates, for every A/C dinuc sequence pair:
- the observed counts of each conf-conf combination  
- the expected frequency of every conf-conf combination, by multiplication of their frequencies  
- the raw histogram of cRMSD values  
- the histogram of cRMSD values, weighted by the observed counts (h1)  
- the histogram of cRMSD values, weighted by the expected frequency (h2)

Then, all h1 are pooled and all h2 are pooled.  
Then, the cRMSD propensity is computed as h1 divided by h2.

Then, the expected frequency P of a conf1-conf2 pair is proportional to F1 x F2 x C, where:  
- Fx is the frequency of confx, estimated as the observed frequency (`freq-{dinuc-seqx}.txt`) among the penta chains  
- C is the cRMSD propensity of the cRMSD between conf1 and conf2

Finally, we have the total number of pair observations O, which is 1/3 of the pairs in all penta chains.  
(every pair XY-YZ contributes to 3 penta chains: XY-YZ-Z?-??, ?X-XY-YZ-Z? and ??-?X-XY-YZ).

Then, we can compute:
- the expected pair count, which is P * O  
- the chance NP of non-observing a pair, which is (1-P)**O

We then build a list of antipairs, which are pairs that are not observed, but that *should* have been observed (NP < 0.01).  
These are stored in `pairstats-{pairseq}-antipairs.txt`.


## collect-antipairs.py
This writes antipair multi-model PDB files as:
- the antipair as two dinucleotide pairs (resid 1,2,100,101) => `antipairs-XXX-nonmerged.pdb`
- the antipair as trinucleotide, with the common nucleotide averaged out => `antipairs-XXX.pdb`
- the same, but with the middle nucleotide aligned onto a reference => `antipairs-XXX-aligned.pdb`


## Synopsis of the pair analysis

Here, we analyze the relation between dinucleotide and trinucleotide fragments.  
At 0.5 A precision, the PDB is 77 % complete; in other words, the primary 0.5A trinucleotide library covers 77 % of the fragments within 0.5.  
An additional 16 % of the fragments is within 1A, i.e. a total of 93 %.

It turns out that this primary library can be described very well (99 % with both fits <0.5A) as *pairs* of primary (i.e. seen in more than one PDB) 0.5A *dinucleotide* fragments.  
These dinucleotide pairs have an excellent compatibility RMSD (97 % < 0.5A).

In conclusion, *primary trinucleotides (that cover 90 % of the cases) can be synthetically generated from superimposed pairs of primary dinucleotides*.

It is then observed that the pairing is *very sparse*: among all potential pairs of primary dinucleotides, only about 1 in 1000 are actually observed.  
Yet, many potential pairs are *plausible*: in terms of compatibility RMSD, 1 out of 18 pairings are under 0.5A.

One could ascribe this sparsity to a simple lack of observations, but statistical analysis shows otherwise:  
a number of pairings between dinucleotide fragments that are common and plausible are nevertheless not observed, and this is shown to be highly statistically significant.

We provide a small "anti-fragment library" of trinucleotides where the probability of absence is <1 % chance to be due to sparse observations.  
This probability of absence was calculated after removing redundancies at the pentanucleotide level, which reduced the number of observations by about half.

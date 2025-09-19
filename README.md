In this repo, the following analyses are performed:

allpdb-rna-fit-dinuc.py
    Create an RNA "monster sequence" (with seg breaks) and fits all library dinucleotides on it.
        The extension library is appended after the primary one.
        The library is filtered for same-PDB-origin (using the replacement library)
    Store as allpdb-rna-fit-dinuc-{motif}.txt
        motifs are A/C/G/U-A/C/G/U
    Format: each line contains the following columns:
        - monster sequence position (starting at 1)
        - best fit library conformer (starting at 1)
        - best fit RMSD
        - is_replacement: if the fragment comes from the replacement library (because the primary conformer had the same PDB origin)

crmsd_matrix.py
    Calculates the compatibility RMSD between each compatible pair of primary dinucleotide libraries.
    The compatibility between AC and CA is stored as crmsd_matrix_ACA.npy

penta.py
   For each unique pentanucleotide sequence, all conformer chains (conf1, conf2, conf3, conf4) are enumerated.
   They are filtered for RMSD<1A for every conf, and only primary confs are considered.
   Terminal chains are padded with X, so that every conf contributes to four chains and every conf pair contibutes to three chains.
   Every conformer chain is stored only once. Non-redundancy is about 55 % of all pentanucleotide conformer chains.

pairstats.py
    Analyses the non-redundant conf chains computed by penta.py
    For the sake of statistics, A/C/G/U => A/C (the conf indices point to A/C libraries)

    Calculates, for every A/C dinuc sequence,
        the observed frequency of every conf  (freq-{dinuc}.txt)

    Calculates, for every A/C dinuc sequence pair:
    - The observed counts of each conf-conf combination
    - The expected frequency of every conf-conf combination, by multiplication of their frequencies
    - The raw histogram of cRMSD values
    - The histogram of cRMSD values, weighted by the observed counts (h1)
    - The histogram of cRMSD values, weighted by the expected frequency (h2)
  Then, all h1 are pooled and all h2 are pooled. Then, the cRMSD propensity is computed as h1 divided by h2.
  
  Then, the expected frequency P of a conf1-conf2 pair is proportional to F1 x F2 x C, where:
    Fx is the frequency of confx, estimated as the observed frequency (freq-{dinuc-seqx}.txt) among the penta chains
    C is the cRMSD propensity of the cRMSD between conf1 and conf2

  Finally, we have the total number of pair observations O, which is 1/3 of the pairs in all penta chains. (every pair XY-YZ contributes to 3 penta chains: XY-YZ-Z?-??, ?X-XY-YZ-Z? and ??-?X-XY-YZ).

  Then, we can compute:
     the expected pair count, which is P * O
     the chance NP of non-observing a pair, which is (1-P)**O

  We the build a list of antipairs, which are pairs that are not observed, but that *should* have been observed (NP < 0.01)
  These are stored in pairstats-{pairseq}-antipairs.txt

collect-antipairs.py
    This writes antipair multi-model PDB files as:
    - The antipair as two dinucleotide pairs (resid 1,2,100,101) => antipairs-XXX-nonmerged.pdb
    - The antipair as trinucleotide, with the common nucleotide averaged out => antipairs-XXX.pdb
    - The same, but with the middle nucleotide aligned onto a reference => antipairs-XXX-aligned.pdb

## Synopsis of the pair analysis

Here, we analyze the relation between dinucleotide and trinucleotide fragments. At 0.5 A precision, the PDB is 77 % complete; in other words, the primary 0.5A trinucleotide library covers 77 % of the fragments within 0.5. An additional 16 % of the fragments is within 1A, i.e. a total of 93 %. It turns out that this primary library can be described very well (REDO! >99 %, even within 0.5A) as *pairs* of primary (i.e. seen in more than one PDB) 0.5A *dinucleotide* fragments. These dinucleotide pairs are REDO! clash-free and have an excellent overlap RMSD (TODO: compatibility RMSD ??) RMSD (90 % < 0.5A, 98.4 % < 0.7A). In conclusion, *primary trinucleotides (that cover 90 % of the cases) can be synthetically generated from superimposed pairs of primary dinucleotides*. It is then observed that the pairing is *very sparse*: among all potential pairs of primary dinucleotides, only about 1 in 7000 are actually observed. Yet, many potential pairs are *plausible*: in terms of compatibility RMSD, 1 out of 18 pairings are under 0.5A, 1 out of 7 are under 0.7A, and internal clashes are essentially absent among all of those. One could ascribe this sparsity to a simple lack of observations, but statistical analysis shows otherwise: a number of pairings between dinucleotide fragments that are common and plausible are nevertheless not observed, and this is shown to be highly statistically significant. (TODO: analysis to see if it is purely explained by higher-order motifs such as pentanucleotides, which you would not expect to see more than once in a random model. TODO: fragment anti-library).  

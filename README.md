# Spoken_Term_Discovery
This demonstration introduces the Spoken Term Discovery task achieved through  Self Organising Map and Context Free grammar. In our approach, 39-dimensional MFCC features are extracted for each frame and exposed to the SOM to produce the cluster label. The labels produced by SOM between two utterances are compared to find the spoken term similarities exist. Further, similar sequences are indexed through CFG.

The above source code demonstrates the step-by-step flow from acoustic feature extraction to similarity matching task. The complete source code was written using python and its native libraries. 

# Citation 

please cite our work:
```
@article{STD2023,
title = {Unsupervised spoken term discovery using pseudo lexical induction},
journal = {International Journal of Speech Technology},
year = {2023},
doi = {https://doi.org/10.1007/s10772-023-10049},
url = {https://rdcu.be/dpAfk},
author = {Sudhakar P and Sreenivasa Rao K and Pabitra Mitra}
}

```

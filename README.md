# Generate to Understand for Representation

This repository contains the code and models discussed in our paper "Generate to Understand for Representation"(https://arxiv.org/abs/2306.10056). 

> Introducing GUR: a pretraining framework that combines language modeling and contrastive learning objectives in a single training step. We select similar text pairs based on their Longest Common Substring (LCS) from raw unlabeled documents and train the model using masked language modeling and unsupervised contrastive learning. The resulting model, GUR, achieves impressive results without any labeled training data, outperforming all other pretrained baselines as a retriever at the recall benchmark in a zero-shot setting. Additionally, GUR maintains its language modeling ability, as demonstrated in our ablation experiment. 


` This code is a little ahead of the paper. `


```
@ARTICLE{2023arXiv230610056X,
       author = {{Xue}, Changshang and {Zhong}, Xiande and {Liu}, Xiaoqing},
        title = "{Generate to Understand for Representation}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language, Computer Science - Information Retrieval, 68T50 (Primary) 03B65, 91F20(Secondary), I.7},
         year = 2023,
        month = jun,
          eid = {arXiv:2306.10056},
        pages = {arXiv:2306.10056},
          doi = {10.48550/arXiv.2306.10056},
archivePrefix = {arXiv},
       eprint = {2306.10056},
 primaryClass = {cs.CL},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023arXiv230610056X},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

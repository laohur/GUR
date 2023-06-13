# Generate to Understand for Representation

This repository contains the code and models discussed in our paper "Generate to Understand for Representation". 
```
Introducing GUR: a pretraining framework that combines language modeling and contrastive learning objectives in a single training step. We select similar text pairs based on their Longest Common Substring (LCS) from raw unlabeled documents and train the model using masked language modeling and unsupervised contrastive learning. The resulting model, GUR, achieves impressive results without any labeled training data, outperforming all other pretrained baselines as a retriever at the recall benchmark in a zero-shot setting. Additionally, GUR maintains its language modeling ability, as demonstrated in our ablation experiment. 

```

> This code is a little ahead of the paper.

> Our code is available at \url{https://github.com/laohur/GUR}.

> Our paper is available at \url{https://github.com/laohur/GUR/blob/main/Generate%20to%20Understand%20for%20Representation.pdf}  [pdf]("./Generate to Understand for Representation.pdf").

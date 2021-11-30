# Information leakage of SER application in Federated Learning
This repository contains the official implementation (in PyTorch) of Attribute Inference Attack of Speech Emotion Recognition in Federated Learning.


## Speech features

We extract a variety of speech representations using OpenSMILE toolkit and pretrained models. You can refer to [OpenSMILE](https://www.audeering.com/research/opensmile/) and [SUPERB](https://arxiv.org/abs/2105.01051) paper for more information.

Below is a listed of features that we include in the current experiment:

Publication Date | Model | Name | Paper | Input | Stride | Pre-train Data | Official Repo 
|---|---|---|---|---|---|---|---
--- | --- | --- | [MM'10](https://dl.acm.org/doi/10.1145/1873951.1874246) | Speech | --- | --- | [EmoBase](https://www.audeering.com/research/opensmile/)
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [APC](https://github.com/Alexander-H-Liu/NPC)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [NPC](https://github.com/Alexander-H-Liu/NPC)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | [NPC](https://github.com/Alexander-H-Liu/NPC)
Dec 11 2020 | DeCoAR 2.0 | decoar2 | [arxiv](https://arxiv.org/abs/2012.06659) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | [speech-representations](https://github.com/awslabs/speech-representations)

```

\begin{tabular}{ccccccccccccc}
    
    \toprule
    & 
    \multicolumn{6}{c}{\textbf{FedSGD}} &
    \multicolumn{6}{c}{\textbf{FedAvg}}
    \rule{0pt}{2.25ex} \\ \cmidrule(lr){2-7} \cmidrule(lr){8-13}
    
    
    \textbf{Speech Feature} & 
    \multicolumn{2}{c}{\textbf{IEMOCAP}$\mathbf{(D_{p})}$} &
    \multicolumn{2}{c}{\textbf{CREMA-D}$\mathbf{(D_{p})}$} &
    \multicolumn{2}{c}{\textbf{MSP-Improv}$\mathbf{(D_{p})}$} &
    \multicolumn{2}{c}{\textbf{IEMOCAP}$\mathbf{(D_{p})}$} &
    \multicolumn{2}{c}{\textbf{CREMA-D}$\mathbf{(D_{p})}$} &
    \multicolumn{2}{c}{\textbf{MSP-Improv}$\mathbf{(D_{p})}$}
    \rule{0pt}{2.25ex} \\ % \cline{2-13}
    
    
    &
    {\textbf{Acc}} & 
    {\textbf{UAR}} &
    {\textbf{Acc}} & 
    {\textbf{UAR}} &
    {\textbf{Acc}} & 
    {\textbf{UAR}} &
    {\textbf{Acc}} & 
    {\textbf{UAR}} &
    {\textbf{Acc}} & 
    {\textbf{UAR}} &
    {\textbf{Acc}} & 
    {\textbf{UAR}}
    \rule{0pt}{2.25ex} \\ \cmidrule(lr){1-1} \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11} \cmidrule(lr){12-13}
    
    \textbf{Emo-Base} &
    63.28\% & 
    61.96\% & 
    70.57\% & 
    61.86\% & 
    54.14\% & 
    46.01\% & 
    61.42\% & 
    61.65\% & 
    71.12\% & 
    64.41\% & 
    53.42\% & 
    47.35\% 
    \rule{0pt}{2.25ex} \\
    
    \textbf{APC} &
    63.72\% & 
    $\mathbf{63.56\%}$ & 
    68.88\% & 
    64.15\% & 
    54.70\% & 
    47.86\% & 
    $\mathbf{64.81\%}$ & 
    $\mathbf{62.97\%}$ & 
    72.35\% & 
    64.39\% & 
    56.69\% & 
    50.26\% 
    \rule{0pt}{2.25ex} \\
    
    \textbf{Vq-APC} &
    61.44\% & 
    59.57\% & 
    70.96\% & 
    64.67\% & 
    55.97\% & 
    49.31\% & 
    62.20\% & 
    61.53\% & 
    74.02\% & 
    67.48\% & 
    56.88\% & 
    48.83\% 
    \rule{0pt}{2.25ex} \\
    
    \textbf{NPC} &
    61.93\% & 
    59.75\% & 
    69.31\% & 
    60.59\% & 
    53.89\% & 
    45.91\% & 
    61.81\% & 
    55.14\% & 
    71.25\% & 
    63.26\% & 
    53.43\% & 
    42.85\% 
    \rule{0pt}{2.25ex} \\
    
    \textbf{DeCoAR 2.0} &
    $\mathbf{63.99\%}$ & 
    62.88\% & 
    72.27\% & 
    65.62\% & 
    56.69\% & 
    49.54\% & 
    62.93\% & 
    61.71\% & 
    73.61\% & 
    65.00\% & 
    56.46\% & 
    50.49\% 
    \rule{0pt}{2.25ex} \\
    
    \textbf{Tera} &
    63.25\% & 
    60.65\% & 
    $\mathbf{72.28\%}$ & 
    $\mathbf{66.19\%}$ & 
    $\mathbf{57.49\%}$ & 
    $\mathbf{52.13\%}$ & 
    64.34\% & 
    62.38\% & 
    $\mathbf{74.32\%}$ & 
    $\mathbf{67.12\%}$ & 
    $\mathbf{57.54\%}$ & 
    $\mathbf{52.37\%}$
    \rule{0pt}{2.25ex} \\ \bottomrule
    
    
\end{tabular}
    
```


## Referecences


**[OpenSMILE](https://www.audeering.com/research/opensmile/)**
```
@inproceedings{eyben2010opensmile,
  title={Opensmile: the munich versatile and fast open-source audio feature extractor},
  author={Eyben, Florian and W{\"o}llmer, Martin and Schuller, Bj{\"o}rn},
  booktitle={Proceedings of the 18th ACM international conference on Multimedia},
  pages={1459--1462},
  year={2010}
}
```

**[SUPERB](https://arxiv.org/abs/2105.01051)**

```
@inproceedings{yang21c_interspeech,
  author={Shu-wen Yang and Po-Han Chi and Yung-Sung Chuang and Cheng-I Jeff Lai and Kushal Lakhotia and Yist Y. Lin and Andy T. Liu and Jiatong Shi and Xuankai Chang and Guan-Ting Lin and Tzu-Hsien Huang and Wei-Cheng Tseng and Ko-tik Lee and Da-Rong Liu and Zili Huang and Shuyan Dong and Shang-Wen Li and Shinji Watanabe and Abdelrahman Mohamed and Hung-yi Lee},
  title={{SUPERB: Speech Processing Universal PERformance Benchmark}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1194--1198},
  doi={10.21437/Interspeech.2021-1775}
}
```
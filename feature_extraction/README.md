### OpenSMILE features
We extract speech features of human knowledge using OpenSMILE toolkit. You can refer to [OpenSMILE](https://www.audeering.com/research/opensmile/) for more information.

### Pretrained features

We extract a variety of deep speech representations using pretrained models. You can refer to [SUPERB](https://arxiv.org/abs/2105.01051) paper for their model architures and pre-training loss styles.

Publication Date | Model | Name | Paper | Input | Stride | Pre-train Data | Official Ckpt | Official Repo 
|---|---|---|---|---|---|---|---|---
5 Apr 2019 | APC | apc | [arxiv](https://arxiv.org/abs/1904.03240) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [APC](https://github.com/Alexander-H-Liu/NPC)
17 May 2020 | VQ-APC | vq_apc | [arxiv](https://arxiv.org/abs/2005.08392) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | O | [NPC](https://github.com/Alexander-H-Liu/NPC)
25 Oct 2019 | Mockingjay | mockingjay | [arxiv](https://arxiv.org/abs/1910.12638) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
12 Jul 2020 | TERA | tera | [arxiv](https://arxiv.org/abs/2007.06028) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/andi611/Self-Supervised-Speech-Pretraining-and-Representation-Learning)
1 Nov 2020 | NPC | npc | [arxiv](https://arxiv.org/abs/2011.00406) | Mel | 10ms | [LibriSpeech-360](http://www.openslr.org/12) | X | [NPC](https://github.com/Alexander-H-Liu/NPC)
Jun 14 2021 | HuBERT | hubert / hubert_large_ll60k | [arxiv](https://arxiv.org/abs/2106.07447) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [Fairseq](https://github.com/pytorch/fairseq)
Dec 3 2019 | DeCoAR | decoar | [arxiv](https://arxiv.org/abs/1912.01679) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [speech-representations](https://github.com/awslabs/speech-representations)
Dec 11 2020 | DeCoAR 2.0 | decoar2 | [arxiv](https://arxiv.org/abs/2012.06659) | Mel | 10ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [speech-representations](https://github.com/awslabs/speech-representations)
Oct 5 2021 | DistilHuBERT | distilhubert | [arxiv](https://arxiv.org/abs/2110.01900) | wav | 20ms | [LibriSpeech-960](http://www.openslr.org/12) | O | [S3PRL](https://github.com/s3prl/s3prl)
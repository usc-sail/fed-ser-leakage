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

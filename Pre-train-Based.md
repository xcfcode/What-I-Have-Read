# Pre-train Based

From https://github.com/RUCAIBox/PLMPapers#survey
For tracking purpose.

# PLM papers

![](https://img.shields.io/github/last-commit/wxl1999/PLMPapers?color=blue) ![](https://img.shields.io/badge/PaperNumber-163-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)

Large-scale pre-trained language models (PLMs) such as BERT and GPT have achieved great success and become a milestone in NLP.

In this repo, we collect some representative PLM papers in recent years based on the number of citations.

We will keep the repo updated and welcome pull requests and issues! Thanks for your stars and forks!

## Table of Contents
- [Survey](#survey)
- [Benchmark](#benchmark)
- [PLM Design](#plm-design)
  - [General](#general)
  - [Knowledge](#knowledge)
  - [Multilingual](#multilingual)
  - [Multi-Modal](#multi-modal)
  - [Information Retrieval](#information-retrieval)
- [PLM Analysis](#plm-analysis)
  - [Knowledge](#knowledge-1)
  - [Robustness](#robustness)
  - [Sparsity](#sparsity)
  - [Others](#others)
- [Efficient PLM](#efficient-plm)
  - [Training](#training)
  - [Compression](#compression)
- [PLM Adaptation](#plm-adaptation)
  - [Two-Stage](#two-stage)
  - [Multi-Task](#multi-task)
  - [Adapater](#adapater)
  - [Prompt](#prompt)
  - [Others](#others-1)

## Survey

* [ ] 1. "Pre-trained models for natural language processing: A survey". `Science China Technological Sciences(2020)` [[PDF]](https://www.sciengine.com/publisher/scp/journal/SCTS/63/10/10.1007/s11431-020-1647-3?slug=fulltext)
* [ ] 2. "Which *BERT? A Survey Organizing Contextualized Encoders". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.608.pdf)
* [ ] 3. "A Primer in BERTology: What We Know About How BERT Works". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.54.pdf)
* [ ] 4. "From static to dynamic word representations: a survey". `International Journal of Machine Learning and Cybernetics(2020)` [[PDF]](http://ir.hit.edu.cn/~car/papers/icmlc2020-wang.pdf)
* [ ] 5. "Overview of the Transformer-based Models for NLP Tasks". `2020 15th Conference on Computer Science and Information Systems (FedCSIS)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9222960)
* [ ] 6. "A Survey on Contextual Embeddings". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2003.07278)
* [ ] 7. "The NLP Cookbook: Modern Recipes for Transformer Based Deep Learning Architectures". `IEEE Access(2021)` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9422763)
* [ ] 8. "Pre-Trained Models: Past, Present and Future". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2106.07139)
* [ ] 9. "A Survey of Transformers". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2106.04554)

## Benchmark

* [ ] 1. **XNLI**: "XNLI: Evaluating Cross-lingual Sentence Representations". `EMNLP(2018)` [[PDF]](https://aclanthology.org/D18-1269.pdf) [[Dataset]](https://github.com/facebookresearch/XNLI)
* [ ] 2. **GLUE**: "GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding". `ICLR(2019)` [[Homepage]](https://gluebenchmark.com/)
* [ ] 3. **SuperGLUE**: "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems". `NeurIPS(2019)` [[Homepage]](https://super.gluebenchmark.com/)
* [ ] 4. **CLUE**: "CLUE: A Chinese Language Understanding Evaluation Benchmark". `COLING(2020)` [[Homepage]](https://www.cluebenchmarks.com/)
* [ ] 5. **XTREME**: "XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization". `ICML(2020)` [[Homepage]](https://sites.research.google/xtreme)
* [ ] 6. **XGLUE**: "XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation". `EMNLP(2020)` [[Homepage]](https://microsoft.github.io/XGLUE/)
* [ ] 7. **DialoGLUE**: "DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue". `arXiv(2020)` [[Homepage]](https://eval.ai/web/challenges/challenge-page/708/overview)

## PLM Design

### General

* [ ] 1. **GPT**: "Improving Language Understanding by Generative Pre-Training". `OpenAI(2018)` [[Project]](https://openai.com/blog/language-unsupervised/)
* [ ] 2. **GPT-2**: "Language Models are Unsupervised Multitask Learners". `OpenAI(2019)` [[Project]](https://openai.com/blog/better-language-models/)
* [ ] 3. **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". `NAACL(2019)` [[PDF]](https://arxiv.org/pdf/1810.04805.pdf) [[Code]](https://github.com/google-research/bert)
* [ ] 4. **XLNet**: "XLNet: Generalized Autoregressive Pretraining for Language Understanding". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) [[Code]](https://github.com/zihangdai/xlnet)
* [ ] 5. **SBERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". `ACL(2019)` [[PDF]](https://aclanthology.org/D19-1410.pdf) [[Code]](https://github.com/UKPLab/sentence-transformers)
* [ ] 6. **UniLM**: "Unified Language Model Pre-training for Natural Language Understanding and Generation". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf) [[Code]](https://github.com/microsoft/unilm)
* [ ] 7. **MASS**: "MASS: Masked Sequence to Sequence Pre-training for Language Generation". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/song19d/song19d.pdf) [[Code]](https://github.com/microsoft/MASS)
* [ ] 8.  **Chinese-BERT-wwm**: "Pre-Training with Whole Word Masking for Chinese BERT". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1906.08101.pdf) [[Code]](https://github.com/ymcui/Chinese-BERT-wwm)
* [ ] 9.  "Cloze-driven Pretraining of Self-attention Networks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1539.pdf)
* [ ] 10. "BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model". `Workshop on Methods for Optimizing and Evaluating Neural Language Generation(2019)` [[PDF]](https://aclanthology.org/W19-2304.pdf) [[Code]](https://github.com/nyu-dl/bert-gen)
* [ ] 11. **GPT-3**: "Language Models are Few-Shot Learners". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2005.14165) [[Code]](https://github.com/openai/gpt-3)
* [ ] 12. **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". `JMLR(2020)` [[PDF]](https://jmlr.org/papers/volume21/20-074/20-074.pdf) [[Code]](https://github.com/google-research/text-to-text-transfer-transformer)
* [ ] 13. **BART**: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.703.pdf) [[Code]](https://github.com/pytorch/fairseq)
* [ ] 14. **Poly-encoders**: "Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SkxgnnNFvH)
* [ ] 15. **SpanBERT**: "SpanBERT: Improving Pre-training by Representing and Predicting Spans". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.5.pdf) [[Code]](https://github.com/facebookresearch/SpanBERT)
* [ ] 16. **ERNIE 2.0**: "ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6428/6284) [[Code]](https://github.com/PaddlePaddle/ERNIE)
* [ ] 17. **SemBERT**: "Semantics-Aware BERT for Language Understanding". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6510/6366) [[Code]](https://github.com/cooelf/)
* [ ] 18. "Leveraging Pre-trained Checkpoints for Sequence Generation Tasks". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.18.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/bertseq2seq)
* [ ] 19. **ProphetNet**: "ProphetNet: Predicting Future N-gram for Sequence-to-SequencePre-training". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.217.pdf)
* [ ] 20. **UniLMv2**: "UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training". `ICML(2020)` [[PDF]](http://proceedings.mlr.press/v119/bao20a/bao20a.pdf) [[Code]](https://github.com/microsoft/unilm)
* [ ] 21. **MacBERT**: "Revisiting Pre-Trained Models for Chinese Natural Language Processing". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.58.pdf) [[Code]](https://github.com/ymcui/MacBERT)
* [ ] 22. **MPNet**: "MPNet: Masked and Permuted Pre-training for Language Understanding". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2004.09297) [[Code]](https://github.com/microsoft/MPNet)
* [ ] 23. **DEBERTA**: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=XPZIaotutsD) [[Code]](https://github.com/microsoft/DeBERTa)
* [ ] 24. **PALM**: "PALM: Pre-training an Autoencoding&Autoregressive Language Model for Context-conditioned Generation". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.700.pdf)

### Knowledge

* [ ] 1. **ERNIE(Baidu)**: "ERNIE: Enhanced Representation through Knowledge Integration". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1904.09223) [[Code]](https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE)
* [ ] 2. **KnowBert**: "Knowledge Enhanced Contextual Word Representations". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1005.pdf)
* [ ] 3. **ERNIE(Tsinghua)**: "ERNIE: Enhanced Language Representation with Informative Entities". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1139.pdf) [[Code]](https://github.com/thunlp/ERNIE)
* [ ] 4. **COMET**: "COMET: Commonsense Transformers for Automatic Knowledge Graph Construction". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1470.pdf) [[Code]](https://github.com/atcbosselut/comet-commonsense)
* [ ] 5. **K-BERT**: "K-BERT: Enabling Language Representation with Knowledge Graph". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/5681/5537) [[Code]](https://github.com/autoliuweijie/K-BERT)
* [ ] 6. **WKLM**: "Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=BJlzm64tDH)
* [ ] 7.  **LUKE**: "LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.523.pdf) [[Code]](https://github.com/studio-ousia/luke)
* [ ] 8.  **K-Adapter**: "K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=CLnj31GZ4cI)
* [ ] 9.  **KEPLER**: "KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation". `TACL(2021)` [[PDF]](https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl_a_00360/1923927/tacl_a_00360.pdf) [[Code]](https://github.com/THU-KEG/KEPLER)

### Multilingual

* [ ] 1. **XLM**: "Cross-lingual Language Model Pretraining". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1901.07291) [[Code]](https://github.com/facebookresearch/XLM)
* [ ] 2. "Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond". `TACL(2019)` [[PDF]](https://aclanthology.org/Q19-1038.pdf) [[Code]](https://github.com/facebookresearch/LASER)
* [ ] 3. **UDify**: "75 Languages, 1 Model: Parsing Universal Dependencies Universally". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1279.pdf) [[Code]](https://github.com/hyperparticle/udify)
* [ ] 4. **Unicoder**: "Unicoder: A Universal Language Encoder by Pre-training with Multiple Cross-lingual Tasks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1252.pdf)
* [ ] 5. **XLM-R**: "Unsupervised Cross-lingual Representation Learning at Scale". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.747.pdf)
* [ ] 6. "Multilingual Alignment of Contextual Word Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=r1xCMyBtPS)
* [ ] 7. **mBART**: "Multilingual Denoising Pre-training for Neural Machine Translation". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.47.pdf) [[Code]](https://github.com/pytorch/fairseq/tree/master/examples/mbart)
* [ ] 8. **mT5**: "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.41.pdf) [[Code]](https://goo.gle/mt5-code)
* [ ] 9. **InfoXLM**: "InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-Training". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.280.pdf) [[Code]](https://aka.ms/infoxlm)

### Multi-Modal

* [ ] 1. **ViLBERT**: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks". `NeuralIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf)
* [ ] 2. **LXMERT**: "LXMERT: Learning Cross-Modality Encoder Representations from Transformers". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1514.pdf) [[Code]](https://github.com/airsplay/lxmert)
* [ ] 3. **VideoBERT**: "VideoBERT: A Joint Model for Video and Language Representation Learning" `ICCV(2019)` [[PDF]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Sun_VideoBERT_A_Joint_Model_for_Video_and_Language_Representation_Learning_ICCV_2019_paper.pdf)
* [ ] 4. **MulT**: "Multimodal Transformer for Unaligned Multimodal Language Sequences". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1656.pdf) [[Code]](https://github.com/yaohungt/Multimodal-Transformer)
* [ ] 5. **VisualBERT**: "VisualBERT: A Simple and Performant Baseline for Vision and Language". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1908.03557.pdf)
* [ ] 6. **B2T2**: "Fusion of Detected Objects in Text for Visual Question Answering". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1219.pdf) [[Code]](https://github.com/google-research/language/tree/master/language/question_answering/b2t2)
* [ ] 7. **VL-BERT**: "VL-BERT: Pre-training of Generic Visual-Linguistic Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SygXPaEYvH) [[Code]](https://github.com/jackroos/VL-BERT)
* [ ] 8. **Unicoder-VL**: "Unicoder-VL: A Universal Encoder for Vision and Language by Cross-Modal Pre-Training". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6795/6649)
* [ ] 9. **VLP**: "Unified Vision-Language Pre-Training for Image Captioning and VQA". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/7005/6859) [[Code]](https://github.com/LuoweiZhou/VLP)
* [ ] 10. **UNITER**: "UNITER: UNiversal Image-TExt Representation Learning". `ECCV(2020)` [[PDF]](https://arxiv.org/pdf/1909.11740) [[Code]](https://github.com/ChenRocks/UNITER)
* [ ] 11. **Oscar**: "Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks". `ECCV(2020)` [[PDF]](https://arxiv.org/pdf/2004.06165) [[Code]](https://github.com/microsoft/Oscar)
* [ ] 12. "12-in-1: Multi-Task Vision and Language Representation Learning". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf) [[Code]](https://github.com/facebookresearch/vilbert-multi-task)
* [ ] 13. **ActBERT**: "ActBERT: Learning Global-Local Video-Text Representations". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_ActBERT_Learning_Global-Local_Video-Text_Representations_CVPR_2020_paper.pdf)
* [ ] 14. **VLN**: "Vision-Language Navigation With Self-Supervised Auxiliary Reasoning Tasks". `CVPR(2020)` [[PDF]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhu_Vision-Language_Navigation_With_Self-Supervised_Auxiliary_Reasoning_Tasks_CVPR_2020_paper.pdf)
* [ ] 15. **VILLA**: "Large-Scale Adversarial Training for Vision-and-Language Representation Learning". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2006.06195.pdf) [[Code]](https://github.com/zhegan27/VILLA)
* [ ] 16. **ImageBERT**: "ImageBERT: Cross-modal Pre-training with Large-scale Weak-supervised Image-Text Data". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2001.07966.pdf)
* [ ] 17. **ALIGN**: "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision". `ICML(2021)` [[PDF]](https://arxiv.org/pdf/2102.05918.pdf) 
* [ ] 18. **ClipBERT**: "Less Is More: ClipBERT for Video-and-Language Learning via Sparse Sampling". `CVPR(2021)` [[PDF]](https://openaccess.thecvf.com/content/CVPR2021/papers/Lei_Less_Is_More_ClipBERT_for_Video-and-Language_Learning_via_Sparse_Sampling_CVPR_2021_paper.pdf) [[Code]](https://github.com/jayleicn/ClipBERT)
* [ ] 19. **DALL·E**: "Zero-Shot Text-to-Image Generation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2102.12092.pdf) [[Code]](https://github.com/openai/DALL-E)
* [ ] 20. **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2103.00020.pdf) [[Code]](https://github.com/OpenAI/CLIP)

### Information Retrieval

* [ ] 1. **ORQA**: "Latent Retrieval for Weakly Supervised Open Domain Question Answering". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1612.pdf)
* [ ] 2. **REALM**: "REALM: Retrieval-Augmented Language Model Pre-Training". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.08909)
* [ ] 3. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) [[Code]](https://github.com/huggingface/transformers/blob/master/examples/rag/)
* [ ] 4. **DPR**: "Dense Passage Retrieval for Open-Domain Question Answering". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.550.pdf) [[Code]](https://github.com/facebookresearch/DPR)

## PLM Analysis

### Knowledge

* [ ] 1. "What Does BERT Look at? An Analysis of BERT’s Attention". `BlackBoxNLP(2019)` [[PDF]](https://aclanthology.org/W19-4828.pdf) [[Code]](https://github.com/clarkkev/attention-analysis)
* [ ] 2. "BERT Rediscovers the Classical NLP Pipeline". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1452.pdf)
* [ ] 3. "How Multilingual is Multilingual BERT?". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1493.pdf)
* [ ] 4. "A Structural Probe for Finding Syntax in Word Representations". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1419.pdf) [[Code]](https://github.com/john-hewitt/structural-probes)
* [ ] 5. "Language Models as Knowledge Bases?". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1250.pdf) [[Code]](https://github.com/facebookresearch/LAMA)
* [ ] 6. "What Does BERT Learn about the Structure of Language?". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1356.pdf) [[Code]](https://github.com/ganeshjawahar/)
* [ ] 7. "Linguistic Knowledge and Transferability of Contextual Representations". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1112.pdf)
* [ ] 8. "Assessing BERT's Syntactic Abilities". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1901.05287.pdf) [[Code]](https://github.com/yoavg/bert-syntax)
* [ ] 9. "Probing Neural Network Comprehension of Natural Language Arguments" `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1459.pdf)
* [ ] 10. "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1006.pdf)
* [ ] 11. "Visualizing and Measuring the Geometry of BERT". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.pdf)
* [ ] 12. "Designing and Interpreting Probes with Control Tasks". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1275.pdf)
* [ ] 13. "Open Sesame: Getting inside BERT’s Linguistic Knowledge". `BlackboxNLP(2019)` [[PDF]](https://aclanthology.org/W19-4825.pdf) [[Code]](https://github.com/yongjie-lin/bert-opensesame)
* [ ] 14. "What do you learn from context? Probing for sentence structure in contextualized word representations". `ICLR(2019)` [[PDF]](https://openreview.net/pdf?id=SJzSgnRcKX) [[Code]](https://github.com/jsalt18-sentence-repl/jiant)
* [ ] 15. "Commonsense Knowledge Mining from Pretrained Models". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1109.pdf)
* [ ] 16. "Do NLP Models Know Numbers? Probing Numeracy in Embeddings". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1534.pdf)
* [ ] 17. "On the Cross-lingual Transferability of Monolingual Representations". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.421.pdf)
* [ ] 18. "Cross-Lingual Ability of Multilingual BERT: An Empirical Study". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=HJeT3yrtDr) [[Code]](https://github.com/CogComp/mbert-study)
* [ ] 19. "What BERT Is Not: Lessons from a New Suite of Psycholinguistic Diagnostics for Language Models". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.3.pdf) [[Code]](https://github.com/aetting/lm-diagnostics)
* [ ] 20. "How Much Knowledge Can You Pack Into the Parameters of a Language Model?". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.437.pdf) [[Code]](https://goo.gle/t5-cbqa)
* [ ] 21. "How Can We Know What Language Models Know?". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.28.pdf) [[Code]](https://github.com/jzbjyb/LPAQA)
* [ ] 22. "oLMpics-On What Language Model Pre-training Captures". `TACL(2020)` [[PDF]](https://aclanthology.org/2020.tacl-1.48.pdf) [[Code]](http://github.com/alontalmor/oLMpics)
* [ ] 23. "Information-Theoretic Probing with Minimum Description Length". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.14.pdf) [[Code]](https://github.com/)
* [ ] 24. "Inducing Relational Knowledge from BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6242/6098)
* [ ] 25. **AutoPrompt**: "AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.346.pdf) [[Code]](http://ucinlp.github.io/autoprompt)
* [ ] 26. "Emergent linguistic structure in artificial neural networks trained by self-supervision". `PNAS(2020)` [[PDF]](https://www.pnas.org/content/pnas/117/48/30046.full.pdf)
* [ ] 27. "Evaluating Commonsense in Pre-Trained Language Models". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6523/6379) [[Code]](https://github.com/XuhuiZhou/CATS)
* [ ] 28. "Inducing Relational Knowledge from BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6242/6098)

### Robustness

* [ ] 1. "Universal Adversarial Triggers for Attacking and Analyzing NLP". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1221.pdf) [[Code]](https://github.com/Eric-Wallace/universal-triggers)
* [ ] 2. "Pretrained Transformers Improve Out-of-Distribution Robustness". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.244.pdf) [[Code]](https://github.com/camelop/NLP-Robustness)
* [ ] 3. **BERT-ATTACK**: "BERT-ATTACK: Adversarial Attack Against BERT Using BERT". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.500.pdf) [[Code]](https://github.com/LinyangLee/BERT-Attack)
* [ ] 4. "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6311/6167) [[Code]](https://github.com/jind11/TextFooler)

### Sparsity

* [ ] 1. "Are Sixteen Heads Really Better than One?". `NeurIPS(2019)` [[PDF]](https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf) [[Code]](https://github.com/pmichel31415/are-16-heads-really-better-than-1)
* [ ] 2. "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1580.pdf) [[Code]](https://github.com/lena-voita/the-story-of-heads)
* [ ] 3. "Revealing the Dark Secrets of BERT". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1445.pdf)
* [ ] 4. "The Lottery Ticket Hypothesis for Pre-trained BERT Networks". `NeurIPS(2020)` [[PDF]](https://proceedings.neurips.cc/paper/2020/file/b6af2c9703f203a2794be03d443af2e3-Paper.pdf) [[Code]](https://github.com/VITA-Group/BERT-Tickets)
* [ ] 5. "When BERT Plays the Lottery, All Tickets Are Winning". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.259.pdf) [[Code]](https://github.com/sai-prasanna/bert-experiments)

### Others

* [ ] 1. "Scaling Laws for Neural Language Models". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2001.08361.pdf)
* [ ] 2. "Extracting Training Data from Large Language Models". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2012.07805.pdf) [[Code]](https://github.com/ftramer/LM_Memorization)

## Efficient PLM

### Training

* [ ] 1. **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1907.11692) [[Code]](https://github.com/pytorch/fairseq)
* [ ] 2. "Efficient Training of BERT by Progressively Stacking". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/gong19a/gong19a.pdf) [[Code]](https://github.com/gonglinyuan/StackingBERT)
* [ ] 3. **Megatron-LM**: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1909.08053.pdf) [[Code]](https://github.com/NVIDIA/Megatron-LM)
* [ ] 4. **ELECTRA**: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=r1xMH1BtvB) [[Code]](https://github.com/google-research/electra)
* [ ] 5. "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=Syx4wnEtvH) [[Code]](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py)
* [ ] 6. **GShard**: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2006.16668.pdf)
* [ ] 7. **Admin**: "Understanding the Difficulty of Training Transformers". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.emnlp-main.463.pdf) [[Code]](https://github.com/LiyuanLucasLiu/Transforemr-Clinic)
* [ ] 8. **ZeRO**: "ZeRO: Memory optimizations Toward Training Trillion Parameter Models". `SC20: International Conference for High Performance Computing, Networking, Storage and Analysis` [[PDF]](https://ieeexplore.ieee.org/abstract/document/9355301) [[Code]](https://github.com/microsoft/deepspeed)
* [ ] 9. **Switch Transformers**: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.03961) [[Code]](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)

### Compression

* [ ] 1. **DistilBERT**: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1910.01108) [[Code]](https://github.com/huggingface/transformers)
* [ ] 2. **PKD**: "Patient Knowledge Distillation for BERT Model Compression". `EMNLP(2019)` [[PDF]](https://aclanthology.org/D19-1441.pdf) [[Code]](https://github.com/intersun/PKD-for-BERT-Model-Compression)
* [ ] 3. "Distilling Task-Specific Knowledge from BERT into Simple Neural Networks". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1903.12136.pdf)
* [ ] 4. **Q8BERT**: "Q8BERT: Quantized 8Bit BERT". `5th Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS 2019` [[PDF]](https://arxiv.org/pdf/1910.06188.pdf)
* [ ] 5. **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=H1eA7AEtvS) [[Code]](https://github.com/google-research/ALBERT)
* [ ] 6. **TinyBERT**: "TinyBERT: Distilling BERT for Natural Language Understanding". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.findings-emnlp.372.pdf) [[Code]](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
* [ ] 7. **Layerdrop**: "Reducing Transformer Depth on Demand with Structured Dropout". `ICLR(2020)` [[PDF]](https://openreview.net/pdf?id=SylO2yStDr) [[Code]](https://github.com/pytorch/fairseq/tree/master/examples/layerdrop)
* [ ] 8. **Q-BERT**: "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT". `AAAI(2020)` [[PDF]](https://ojs.aaai.org/index.php/AAAI/article/view/6409/6265)
* [ ] 9. **MobileBERT**: "MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.195.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/mobilebert)
* [ ] 10. "Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning". `5th Workshop on Representation Learning for NLP(2020)` [[PDF]](https://aclanthology.org/2020.repl4nlp-1.18.pdf) [[Code]](https://github.com/mitchellgordon95/bert-prune)
* [ ] 11. **MiniLM**: "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.10957.pdf) [[Code]](https://github.com/microsoft/unilm/tree/master/minilm)
* [ ] 12. **FastBERT**: "FastBERT: a Self-distilling BERT with Adaptive Inference Time". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.537.pdf) [[Code]](https://github.com/autoliuweijie/FastBERT)
* [ ] 13. **DeeBERT**: "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.204.pdf) [[Code]](https://github.com/castorini/DeeBERT)

## PLM Adaptation

### Two-Stage

* [ ] 1. "Sentence Encoders on STILTs: Supplementary Training on Intermediate Labeled-data Tasks". `arXiv(2018)` [[PDF]](https://arxiv.org/pdf/1811.01088.pdf) [[Code]](https://github.com/zphang/pytorch-pretrained-BERT)
* [ ] 2. "How to Fine-Tune BERT for Text Classification?". `CCL(2019)` [[PDF]](http://cips-cl.org/static/anthology/CCL-2019/CCL-19-141.pdf)
* [ ] 3. "Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.740.pdf) [[Code]](https://github.com/allenai/dont-stop-pretraining)
* [ ] 4. "Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work?". `ACL(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.467.pdf)

### Multi-Task

* [ ] 1. **MT-DNN**: "Multi-Task Deep Neural Networks for Natural Language Understanding". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1441.pdf) [[Code]](https://github.com/namisan/mt-dnn)
* [ ] 2. "BAM! Born-Again Multi-Task Networks for Natural Language Understanding". `ACL(2019)` [[PDF]](https://aclanthology.org/P19-1595.pdf) [[Code]](https://github.com/google-research/google-research/tree/master/bam)
* [ ] 3. "Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding". `arXiv(2019)` [[PDF]](https://arxiv.org/pdf/1904.09482.pdf) [[Code]](https://github.com/namisan/mt-dnn)

### Adapater

* [ ] 1. "BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/stickland19a/stickland19a.pdf) [[Code]](https://github.com/AsaCooperStickland/Bert-n-Pals)
* [ ] 2. **Adapter**: "Parameter-Efficient Transfer Learning for NLP". `ICML(2019)` [[PDF]](http://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf) [[Code]](https://github.com/google-research/adapter-bert)

### Prompt

* [ ] 1. **PET**: "Exploiting Cloze-Questions for Few-Shot Text Classification and Natural Language Inference". `EACL(2021)` [[PDF]](https://aclanthology.org/2021.eacl-main.20.pdf) [[Code]](https://github.com/timoschick/pet)
* [ ] 2. "It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners". `NAACL(2021)` [[PDF]](https://aclanthology.org/2021.naacl-main.185.pdf) [[Code]](https://github.com/timoschick/pet)
* [ ] 3. "Prefix-Tuning: Optimizing Continuous Prompts for Generation". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.00190.pdf)
* [ ] 4. **LM-BFF**: "Making Pre-trained Language Models Better Few-shot Learners". `ACL(2021)` [[PDF]](https://arxiv.org/pdf/2012.15723) [[Code]](https://github.com/princeton-nlp/LM-BFF)
* [ ] 5. "What Makes Good In-Context Examples for GPT-3?". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2101.06804.pdf) [[Code]](https://github.com/google-research/language/tree/master/language/totto)
* [ ] 6. "The Power of Scale for Parameter-Efficient Prompt Tuning". `arXiv(2021)` [[PDF]](https://arxiv.org/pdf/2104.08691.pdf)

### Others

* [ ] 1. "To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks". `RepL4NLP(2019)` [[PDF]](https://aclanthology.org/W19-4302.pdf)
* [ ] 2. "An Embarrassingly Simple Approach for Transfer Learning from Pretrained Language Models". `NAACL(2019)` [[PDF]](https://aclanthology.org/N19-1213.pdf) [[Code]](https://github.com/alexandra-chron/siatl)
* [ ] 3. "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping". `arXiv(2020)` [[PDF]](https://arxiv.org/pdf/2002.06305.pdf)
* [ ] 4. **SMART**: "SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization". `EMNLP(2020)` [[PDF]](https://aclanthology.org/2020.acl-main.197.pdf) [[Code]](https://github.com/namisan/mt-dnn)
* [ ] 5. "Revisiting Few-sample BERT Fine-tuning". `ICLR(2021)` [[PDF]](https://openreview.net/pdf?id=cO1IH43yUF)

<!--
[Advanced pre-training language models a brief introduction Slides](slides/presentation/Advanced%20pre-training%20language%20models%20a%20brief%20introduction.pdf)

1. **Knowledge Inheritance for Pre-trained Language Models** *Yujia Qin, Yankai Lin, Jing Yi, Jiajie Zhang, Xu Han, Zhengyan Zhang, Yusheng Su, Zhiyuan Liu, Peng Li, Maosong Sun, Jie Zhou* [[pdf]](https://arxiv.org/abs/2105.13880) [[code]](https://github.com/thunlp/Knowledge-Inheritance)

1. **Syntax-Enhanced Pre-trained Model** *Zenan Xu, Daya Guo, Duyu Tang, Qinliang Su, Linjun Shou, Ming Gong, Wanjun Zhong, Xiaojun Quan, Nan Duan, Daxin Jiang* [[pdf]](https://arxiv.org/abs/2012.14116)
2. **Language Models are Open Knowledge Graphs** *Chenguang Wang, Xiao Liu, Dawn Song* [[pdf]](https://arxiv.org/abs/2010.11967)

## Survey

| Paper | Conference |
| :---: | :---: |
|[Pre-trained Models for Natural Language Processing: A Survey](https://arxiv.org/abs/2003.08271)||

## Knowledge
| Paper | Conference |
| :---: | :---: |
|K-ADAPTER: Infusing Knowledge into Pre-Trained Models with Adapters||
|Pretrained Encyclopedia: Weakly Supervised Knowledge-Pretrained Language Model |ICLR20|
|A Knowledge-Enhanced Pretraining Model for Commonsense Story Generation|TACL19|
|ERNIE: Enhanced Representation through Knowledge Integration||
|ERNIE: Enhanced Language Representation with Informative Entities|ACL19|

## Multi-Lingual
| Paper | Conference |
| :---: | :---: |
|[Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210)||

## Generation
| Paper | Conference |
| :---: | :---: |
|[QURIOUS: Question Generation Pretraining for Text Generation](https://arxiv.org/pdf/2004.11026.pdf)|ACL20|
|BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension||
|Unified Language Model Pre-training for Natural Language Understanding and Generation||
|MASS: Masked Sequence to Sequence Pre-training for Language Generation|ICML19|

| Paper | Conference |
| :---: | :---: |
|[Cross-Thought for Sentence Encoder Pre-training](https://arxiv.org/abs/2010.03652)|EMNLP2020|
|[Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)||
|[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)||
|[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)|ACL20|
|Make Lead Bias in Your Favor : A Simple and Effective Method for News Summarization||
|DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation||
| What Does BERT Look At? An Analysis of BERT’s Attention | EMNLP19 |
|SpanBERT: Improving Pre-training by Representing and Predicting Spans||
|XLNet: Generalized Autoregressive Pretraining for Language Understanding||
|Pre-Training with Whole Word Masking for Chinese BERT||
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding|NAACL19|
|Linguistic Knowledge and Transferability of Contextual Representations|NAACL19|
|Improving Language Understanding by Generative Pre-Training||
|Deep contextualized word representations|NAACL18|
-->



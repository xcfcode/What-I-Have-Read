# Summarization

* [Summarization](#summarization)
* [Slides](#slides)
* [Survey](#survey)
* [Dialogue](#dialogue)
    * [Medical](#medical)
    * [French Meeting](#french-meeting)
    * [Meeting](#meeting)
        * [Multi-modal](#multi-modal)
    * [Open Domain](#open-domain)
    * [Customer Service](#customer-service)
    * [Email](#email)
    * [News Review](#news-review)
    * [Others](#others)
* [Factuality](#factuality)
* [Graph](#graph)
* [Emotion related](#emotion-related)
* [Pre-train Based](#pre-train-based)
* [Style](#style)
* [Multi-Document](#multi-document)
* [Cross-Lingual](#cross-lingual)
* [Unsupervised](#unsupervised)
* [Dataset](#dataset)
* [Multi-modal](#multi-modal-1)
* [Concept-map-based](#concept-map-based)
* [Timeline](#timeline)
* [Opinion](#opinion)
* [Reinforcement Learning](#reinforcement-learning)
* [Sentence Summarization](#sentence-summarization)
* [Evaluation](#evaluation)
* [Discourse](#discourse)
* [Controlled](#controlled)
* [Others](#others-1)

## Slides
* [Multi-modal Summarization](slides/presentation/Multi-modal-Summarization.pdf)
* [ACL20 Summarization](slides/presentation/acl2020-summarization.pdf)
* [Summarization](slides/notes/Brief-intro-to-summarization.pdf)
* [文本摘要简述](slides/presentation/文本摘要简述.pdf)
* [ACL19 Summarization](slides/presentation/ACL19%20Summarization.pdf)
* [EMNLP19 Summarization](slides/notes/EMNLP19_Summarization.pdf)
* [ACL19-A Simple Theoretical Model of Importance for Summarization](slides/paper-slides/A%20Simple%20Theoretical%20Model%20of%20Importance%20for%20Summarization.pdf)
* [ACL19-Multimodal Abstractive Summarization for How2 Videos](slides/paper-slides/Multimodal%20Abstractive%20Summarization%20for%20How2%20Videos.pdf)

## Survey
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[From Standard Summarization to New Tasks and Beyond: Summarization with Manifold Information](https://arxiv.org/pdf/2005.04684.pdf)|IJCAI20||
|[Neural Abstractive Text Summarization with Sequence-to-Sequence Models: A Survey](https://arxiv.org/pdf/1812.02303.pdf)|||
|[A Survey on Neural Network-Based Summarization Methods](https://arxiv.org/pdf/1804.04589.pdf)|||
|[Automated text summarisation and evidence-based medicine: A survey of two domains](https://arxiv.org/pdf/1804.04589.pdf)|||
|[Automatic Keyword Extraction for Text Summarization: A Survey](https://arxiv.org/pdf/1704.03242.pdf)|||
|[Text Summarization Techniques: A Brief Survey](https://arxiv.org/pdf/1707.02268.pdf)|||
|[Recent automatic text summarization techniques: a survey](https://link.springer.com/article/10.1007/s10462-016-9475-9)|||


## Dialogue 

### Medical
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Medical Dialogue Summarization for Automated Reporting in Healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7225507/)|2020|triple extraction, SOAP notes, Pipeline|
|[Alignment Annotation for Clinic Visit Dialogue to Clinical Note Sentence Language Generation](https://www.aclweb.org/anthology/2020.lrec-1.52/)|LREC2020||
|[Generating SOAP Notes from Doctor-Patient Conversations](https://arxiv.org/pdf/2005.01795.pdf)|2020|Doctor-Patient Conversations, SOAP notes, Extractive+Abstractive|
|[Automatically Generating Psychiatric Case Notes From Digital Transcripts of Doctor-Patient Conversations](https://www.aclweb.org/anthology/W19-1918/)|NAACL19|
| Topic-aware Pointer-Generator Networks for Summarizing Spoken Conversations |ASRU19 |

### French Meeting
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation](https://www.aclweb.org/anthology/2020.lrec-1.829)|LREC2020|French public_meetings 22 meetings 1060 pairs|
|[Leverage Unlabeled Data for Abstractive Speech Summarization with Self-Supervised Learning and Back-Summarization](https://arxiv.org/abs/2007.15296)|SPECOM2020|French meeting summarization|

### Meeting
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Meeting Summarization, A Challenge for Deep Learning](https://link.springer.com/chapter/10.1007/978-3-030-20521-8_53)||
|[End-to-End Abstractive Summarization for Meetings](https://arxiv.org/abs/2004.02016)|2020|Meeting|
| [Abstractive Meeting Summarization via Hierarchical Adaptive Segmental Network Learning](https://dl.acm.org/doi/10.1145/3308558.3313619) | WWW19 |
| [Abstractive Dialogue Summarization with Sentence-Gated Modeling Optimized by Dialogue Acts](https://arxiv.org/abs/1809.05715) | SLT18 |
| [Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization](https://arxiv.org/abs/1805.05271) | ACL18|
| [Abstractive Meeting Summarization Using Dependency Graph Fusion](https://arxiv.org/abs/1609.07035) | WWW15 |
| [Text Summarization through Entailment-based Minimum Vertex Cover](https://www.aclweb.org/anthology/S14-1010/)|ENLG13|
|[Domain-Independent Abstract Generation for Focused Meeting Summarization](https://www.aclweb.org/anthology/P13-1137.pdf)|ACL13||
| [Summarizing Decisions in Spoken Meetings](https://arxiv.org/abs/1606.07965) | ACL11 |
|[Automatic analysis of multiparty meetings](https://link.springer.com/article/10.1007/s12046-011-0051-3)|11|
|[A keyphrase based approach to interactive meeting summarization](https://ieeexplore.ieee.org/document/4777863)|08||key phrase guide
|[Focused Meeting Summarization via Unsupervised Relation Extraction](https://www.aclweb.org/anthology/W12-1642.pdf)||
|[Exploring Speaker Characteristics for Meeting Summarization](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_2518.pdf)|10|

#### Multi-modal
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[A Multimodal Meeting Browser that Implements an Important Utterance Detection Model based on Multimodal Information](https://dl.acm.org/doi/abs/10.1145/3379336.3381491)|||
|[Exploring Methods for Predicting Important Utterances Contributing to Meeting Summarization](https://www.mdpi.com/2414-4088/3/3/50)|19|☆|
| [Keep Meeting Summaries on Topic: Abstractive Multi-Modal Meeting Summarization](https://www.aclweb.org/anthology/P19-1210/)| ACL19 |
|[Fusing Verbal and Nonverbal Information for Extractive Meeting Summarization](https://dl.acm.org/doi/10.1145/3279981.3279987)|GIFT18|
|[Meeting Extracts for Discussion Summarization Based on Multimodal Nonverbal Information](https://dl.acm.org/doi/10.1145/2993148.2993160)|ICMI16|
|[Extractive Summarization of Meeting Recordings](https://pdfs.semanticscholar.org/6159/506bdd368fff24dd12e5c6ed91ba05b44f9e.pdf)|2005|
| [Multimodal Summarization of Meeting Recordings](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.862.6509&rep=rep1&type=pdf)|ICME 2003|

### Open Domain
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237)|EMNLP19|Chit-chat|
|[Making Sense of Group Chat through Collaborative Tagging and Summarization](https://homes.cs.washington.edu/~axz/papers/cscw_tilda.pdf)|CSCW18|System, Best paper award, Structured summary by tags and notes|
|[Collabot: Personalized Group Chat Summarization](https://dl.acm.org/doi/10.1145/3159652.3160588)|WSDM18|

### Customer Service
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
| [Automatic Dialogue Summary Generation for Customer Service](https://dl.acm.org/doi/10.1145/3292500.3330683) |KDD19|

### Email
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Building a Dataset for Summarization and Keyword Extraction from Emails](http://www.lrec-conf.org/proceedings/lrec2014/pdf/1037_Paper.pdf)|2014|349 emails and threads 137 threads|
|[Task-focused Summarization of Email](https://www.aclweb.org/anthology/W04-1008.pdf)|2004|

### News Review
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[The SENSEI Annotated Corpus: Human Summaries of Reader Comment Conversations in On-line News](https://www.aclweb.org/anthology/W16-3605/)|SIGDIAL16|

### Others
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset|ACL20|
| Legal Summarization for Multi-role Debate Dialogue via Controversy Focus Mining and Multi-task Learning|CIKM19|
| Abstractive Dialog Summarization with Semantic Scaffolds ||
|[Creating a reference data set for the summarization of discussion forum threads](https://link.springer.com/article/10.1007/s10579-017-9389-4)||
|[Summarizing Dialogic Arguments from Social Media](https://arxiv.org/abs/1711.00092)|SemDial 2017|debate dialogue, extractive|
| Abstractive Summarization of Spoken and Written Conversation | arxiv |
| Dial2Desc: End-to-end Dialogue Description Generation | arxiv |
|[Using Summarization to Discover Argument Facets in Online Idealogical Dialog](https://www.aclweb.org/anthology/N15-1046.pdf)|NAACL15|Argumentative Dialogue Summary Corpus|
|Plans Toward Automated Chat Summarization|ACL11|

## Factuality 
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization](https://arxiv.org/abs/2008.11293)|||
|[Fact-based Content Weighting for Evaluating Abstractive Summarisation](https://www.aclweb.org/anthology/2020.acl-main.455/)|ACL20|
|[On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661)|ACL20||
|[Improving Truthfulness of Headline Generation](https://arxiv.org/abs/2005.00882)|ACL20||
|[Knowledge Graph-Augmented Abstractive Summarization with Semantic-Driven Cloze Reward](https://arxiv.org/abs/2005.01159)|ACL20||
|[FEQA : A Question Answering Evaluation Framework for Faithfulness Assessment in Abstractive Summarization](https://arxiv.org/abs/2005.03754)|ACL20||
|[Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports](https://arxiv.org/abs/1911.02541)|ACL20||
|[Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228)|ACL20|* Question generation(BART, NewsQA), * Question Answering(BERT, SQuAD2.0), * Answer similarity(Token level F1)|
|[Boosting Factual Correctness of Abstractive Summarization with Knowledge Graph](https://arxiv.org/abs/2003.08612)||Information Extraction --> Local knowledge Graph|
|[Ranking Generated Summaries by Correctness : An Interesting but Challenging Application for Natural Language Inference](https://www.aclweb.org/anthology/P19-1213/)|ACL19||
|[Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)|||
|[Assessing The Factual Accuracy of Generated Text](https://arxiv.org/abs/1905.13322)|KDD19||
|[Faithful to the Original: Fact Aware Neural Abstractive Summarization](https://arxiv.org/abs/1711.04434)|AAAI18|Information Extraction (openIE+dependency) + Generation|
|[Ensure the Correctness of the Summary : Incorporate Entailment Knowledge into Abstractive Sentence Summarization](https://www.aclweb.org/anthology/C18-1121/)|COLING18||
|[Mind The Facts: Knowledge-Boosted Coherent Abstractive Text Summarization](https://www.microsoft.com/en-us/research/uploads/prod/2019/10/Fact_Aware_Abstractive_Text_Summarization.pdf)|NeurIPS 2019 KR2ML workshop|

## Graph
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Heterogeneous Graph Neural Networks for Extractive Document Summarization](https://arxiv.org/abs/2004.12393)|ACL20|Word-TFIDF-Sentence Graph (GAT)|

## Emotion related
| Paper | Conference | 
| :---: | :---: |
|[A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification](https://arxiv.org/abs/1805.01089)|IJCAI18|
|[Two-level Text Summarization from Online News Sources with Sentiment Analysis](Two-level text summarization from online news sources with sentiment analysis)|IEEE17|
|Creating Video Summarization From Emotion Perspective|ICSP16|

## Pre-train Based
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[QURIOUS: Question Generation Pretraining for Text Generation](https://arxiv.org/pdf/2004.11026.pdf)|ACL20||
|[PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)|ICML20||
|[STEP: Sequence-to-Sequence Transformer Pre-training for Document Summarization](https://arxiv.org/abs/2004.01853)||
|[Abstractive Text Summarization based on Language Model Conditioning and Locality Modeling](https://arxiv.org/abs/2003.13027)||
|[Abstractive Summarization with Combination of Pre-trained Sequence-to-Sequence and Saliency Models](https://arxiv.org/abs/2003.13028)||
|Learning by Semantic Similarity Makes Abstractive Summarization Better||
|TED:A PRETRAINED UNSUPERVISED SUMMARIZATION MODEL WITH THEME MODELING AND DENOISING||
| Make Lead Bias in Your Favor- A Simple and Effective Method for News Summarization |ICLR20 under review |
| Text Summarization with Pretrained Encoders | EMNLP19 |
| HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization | ACL19 |
| MASS: Masked Sequence to Sequence Pre-training for Language Generation|ICML19|
| Pretraining-Based Natural Language Generation for Text Summarization||
| Fine-tune BERT for Extractive Summarization||
| Unified Language Model Pre-training for Natural Language Understanding and Generation |NIPS19|
| Self-Supervised Learning for Contextualized Extractive Summarization |ACL19 |
| Efficient Adaptation of Pretrained Transformers for Abstractive Summarization ||

## Style
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[Hooks in the Headline: Learning to Generate Headlines with Controlled Styles](https://arxiv.org/abs/2004.01980)|ACL20|Summarization + Style transfer|

## Multi-Document
| Paper | Conference |
| :---: | :---: |
|[SUPERT: Towards New Frontiers in Unsupervised Evaluation Metrics for Multi-Document Summarization](https://arxiv.org/abs/2005.03724)|ACL20||
|[Leveraging Graph to Improve Abstractive Multi-Document Summarization](https://arxiv.org/abs/2005.10043)|ACL20|
|GameWikiSum : a Novel Large Multi-Document Summarization Dataset|LREC20|
|Generating Representative Headlines for News Stories|WWW20|
| Learning to Create Sentence Semantic Relation Graphs for Multi-Document Summarization | EMNLP19 |
| Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization|ACL19|
| Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model | ACL19 |
| Hierarchical Transformers for Multi-Document Summarization | ACL19 |
| MeanSum : A Neural Model for Unsupervised Multi-Document Abstractive Summarization|ICML19|
| Generating Wikipedia By Summarizing Long Sequence | ICLR18 |
| Adapting the Neural Encoder-Decoder Framework from Single to Multi-Document Summarization|EMNLP18|
| Graph-based Neural Multi-Document Summarization|CoNLL17|
| Improving Multi-Document Summarization via Text Classification | AAAI17|
|Bringing Structure into Summaries : Crowdsourcing a Benchmark Corpus of Concept Maps|EMNLP17|
| An Unsupervised Multi-Document Summarization Framework Based on Neural Document Model | COLING16 |
| 基于文档语义图的中文多文档摘要生成机制|(2009)|
| Event-Centric Summary Generation| (2004) |



## Cross-Lingual
| Paper | Conference |
| :---: | :---: |
|[A Deep Reinforced Model for Zero-Shot Cross-Lingual Summarization with Bilingual Semantic Similarity Rewards](https://www.aclweb.org/anthology/2020.ngt-1.7/)|ACL20 workshop|
|[Jointly Learning to Align and Summarize for Neural Cross-Lingual Summarization](https://www.aclweb.org/anthology/2020.acl-main.554.pdf)|ACL20|
|[Attend, Translate and Summarize:An Efficient Method for Neural Cross-Lingual Summarization](https://www.aclweb.org/anthology/2020.acl-main.121.pdf)|ACL20|
|[MultiSumm: Towards a Unified Model for Multi-Lingual Abstractive Summarization](https://aaai.org/Papers/AAAI/2020GB/AAAI-CaoY.7050.pdf)|AAAI20|
| [Global Voices: Crossing Borders in Automatic News Summarization](https://arxiv.org/abs/1910.00421) | EMNLP19 |
| [NCLS: Neural Cross-Lingual Summarization](https://arxiv.org/abs/1909.00156) | EMNLP19|
| [Zero-Shot Cross-Lingual Abstractive Sentence Summarization through Teaching Generation and Attention](https://www.aclweb.org/anthology/P19-1305/) | ACL19 |
| [A Robust Abstractive System for Cross-Lingual Summarization](https://www.aclweb.org/anthology/N19-1204/)|NAACL19|
|[Cross-Lingual Korean Speech-to-Text Summarization](https://link.springer.com/chapter/10.1007/978-3-030-14799-0_17)|ACIIDS19|
| [Zero-Shot Cross-Lingual Neural Headline Generation](https://dl.acm.org/doi/10.1109/TASLP.2018.2842432)|IEEE/ACM TRANSACTIONS 18|
|[Cross-Language Text Summarization using Sentence and Multi-Sentence Compression](https://hal.archives-ouvertes.fr/hal-01779465/document)|NLDB18|
|Cross-language document summarization based on machine translation quality prediction|ACL10|
|Using bilingual information for cross-language document summarization|ACL11|
|Phrase-based Compressive Cross-Language Summarization|ACL15|
|[Abstractive Cross-Language Summarization via Translation Model Enhanced Predicate Argument Structure Fusing](http://www.nlpr.ia.ac.cn/cip/ZhangPublications/zhang-taslp-2016.pdf)|IEEE/ACM Trans16|
|[Multilingual Single-Document Summarization with MUSE](https://www.aclweb.org/anthology/W13-3111/)|MultiLing13|
|Cross-language document summarization via extraction and ranking of multiple summaries||


## Unsupervised
| Paper | Conference |
| :---: | :---: |
|TED:A PRETRAINED UNSUPERVISED SUMMARIZATION MODEL WITH THEME MODELING AND DENOISING||
| Abstractive Document Summarization without Parallel Data ||
| Unsupervised Neural Single-Document Summarization of Reviews via Learning Latent Discourse Structure and its Ranking | ACL19 |
| MeanSum : A Neural Model for Unsupervised Multi-Document Abstractive Summarization|ICML19|
|SEQ 3 : Differentiable Sequence-to-Sequence-to-Sequence Autoencoder for Unsupervised Abstractive Sentence Compression|NAACL19|
|Learning to Encode Text as Human-Readable Summaries usingGenerative Adversarial Networks|EMNLP18|
|Unsupervised Abstractive Meeting Summarization with Multi-Sentence Compression and Budgeted Submodular Maximization|ACL18|


<!--## Dataset
| Paper | Conference |
| :---: | :---: |
|[MATINF: A Jointly Labeled Large-Scale Dataset for Classiﬁcation, Question Answering and Summarization](https://arxiv.org/abs/2004.12302)|ACL20|
|[Learning to Summarize Passages: Mining Passage-Summary Pairs from Wikipedia Revision Histories](https://arxiv.org/abs/2004.02592)||-->
<!--|GameWikiSum : a Novel Large Multi-Document Summarization Dataset|LREC20|-->
<!--|Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization|EMNLP19|-->
<!--|Learning towards Abstractive Timeline Summarization|IJCAI19|-->
<!--| auto-hMDS: Automatic Construction of a Large Heterogeneous Multilingual Multi-Document Summarization Corpus | |
| Multi-News: a Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model | ACL19 |
| NEWSROOM: A Dataset of 1.3 Million Summaries with Diverse Extractive Strategies||-->
<!--| TALKSUMM: A Dataset and Scalable Annotation Method for Scientiﬁc Paper Summarization Based on Conference Talks | ACL19 |-->
<!--| BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization  | ACL19 |-->
<!--| ScisummNet: A Large Annotated Corpus and Content-Impact Models for Scientiﬁc Paper Summarization with Citation Networks | AAAI19 |-->
<!--| Abstractive Summarization of Reddit Posts with Multi-level Memory Networks|NAACL19|-->
<!--| WikiHow: A Large Scale Text Summarization Dataset ||-->
<!--| Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization|EMNLP18|-->
<!--| How2: A Large-scale Dataset for Multimodal Language Understanding|NIPS18|-->
<!--| Abstractive Text Summarization by Incorporating Reader Comments|AAAI19|-->
<!--| Generating Wikipedia By Summarizing Long Sequence | ICLR18 |-->
<!--|A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents|NAACL18|-->
<!--|Bringing Structure into Summaries : Crowdsourcing a Benchmark Corpus of Concept Maps|EMNLP17|-->

## Dataset
|ID|Name|Description|Paper|Conference|Highlights|
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | CNN\-DailyMail | 新闻领域摘要| Abstractive Text Summarization using Sequence\-to\-sequence RNNs and Beyond |
| 2 | New York Times| 新闻领域摘要 | The New York Times Annotated Corpus |
| 3 | DUC| 新闻领域摘要 | The Effects Of Human Variation In DUC Summarization Evaluation |
| 4 | Gigaword | 新闻领域摘要 | A Neural Attention Model For Abstractive Sentence Summarization |
| 5 | Newsroom | 新闻领域摘要 | Newsroom: A Dataset of 1\.3 Million Summaries with Diverse Extractive Strategies|
| 6 | Xsum | 新闻领域摘要 | Don’t Give Me the Details, Just the Summary\! Topic\-Aware Convolutional Neural Networks for Extreme Summarization|EMNLP18|
| 7 | Multi\-News| 新闻领域多文档摘要| Multi\-News: a Large\-Scale Multi\-Document Summarization Dataset and Abstractive Hierarchical Model|ACL19|
| 8 | SAMSum| 多人闲聊摘要 | [SAMSum Corpus: A Human\-annotated Dialogue Dataset for Abstractive Summarization](https://arxiv.org/abs/1911.12237)|EMNLP19|
| 9 | AMI | 会议摘要 | The AMI Meeting Corpus: A pre\-announcement\. |
| 10 | ICSI| 会议摘要 | The ICSI Meeting Corpus |
| 11 | MSMO| 图片多模态摘要| [MSMO: Multimodal Summarization with Multimodal Output](https://www.aclweb.org/anthology/D18-1448/) |EMNLP18|
| 12 | How2 | 视频多模态摘要| How2: A Large\-scale Dataset for Multimodal Language Understanding| NIPS18|
| 13 | ScisummNet | 科学文献摘要 | ScisummNet: A Large Annotated Corpus and Content\-Impact Models for Scientific Paper Summarization with Citation Networks |AAAI19|
| 14 | PubMed, ArXiv | 科学文献摘要 | A Discourse\-Aware Attention Model for Abstractive Summarization of Long Documents | NAACL18 |
| 15 | TALKSUMM | 科学文献摘要 | TALKSUMM: A Dataset and Scalable Annotation Method for Scientiﬁc Paper Summarization Based on Conference Talks | ACL19 |
| 16 | BillSum | 法律领域摘要 | [BillSum: A Corpus for Automatic Summarization of US Legislation](https://www.aclweb.org/anthology/D19-5406/) |EMNLP19|
| 17 | LCSTS| 中文微博短文本摘要| [LCSTS: A Large Scale Chinese Short Text Summarization Dataset ](https://www.aclweb.org/anthology/D15-1229/)|EMNLP15|
| 18 | WikiHow| 在线知识库领域摘要| WikiHow: A Large Scale Text Summarization Dataset |
| 19 | Concept\-map\-based MDS Corpus| 教育领域多文档摘要| Bringing Structure into Summaries : Crowdsourcing a Benchmark Corpus of Concept Maps|EMNLP17|
| 20 | WikiSum | 维基百科领域多文档摘要| Generating Wikipedia By Summarizing Long Sequence |ICLR18|
| 21 | GameWikiSum | 游戏领域多文档摘要| GameWikiSum : a Novel Large Multi\-Document Summarization Dataset |LREC20|
| 22 | En2Zh CLS, Zh2En CLS| 英文到中文、中文到英语跨语言摘要 | [NCLS: Neural Cross\-Lingual Summarization](https://arxiv.org/abs/1909.00156) |EMNLP19|
| 23 | Timeline Summarization Dataset| 百度百科人物领域时间线摘要| Learning towards Abstractive Timeline Summarization |IJCAI19|
| 24 | Reddit TIFU | 在线讨论领域摘要 | Abstractive Summarization of Reddit Posts with Multi\-level Memory Networks| NAACL19 |
| 25 | TripAtt | 评论领域摘要 | [Attribute\-aware Sequence Network for Review Summarization](https://www.aclweb.org/anthology/D19-1297/)|EMNLP19|
| 26 | Reader Comments Summarization Corpus | 基于评论的微博文本摘要| Abstractive Text Summarization by Incorporating Reader Comments |AAAI19|
| 27 | BIGPATENT | 专利领域生成式摘要| BIGPATENT: A Large\-Scale Dataset for Abstractive and Coherent Summarization|ACL19|
| 28 | Curation Corpus | 新闻领域生成式摘要| [Curation Corpus for Abstractive Text Summarisation](https://github.com/CurationCorp/curation-corpus) |
| 29 | MATINF |多任务文本分类，问答，摘要|[MATINF: A Jointly Labeled Large-Scale Dataset for Classification, Question Answering and Summarization](https://arxiv.org/abs/2004.12302)|ACL20|
| 30 | MLSUM |Multi-Lingual Summarization Dataset|[MLSUM: The Multilingual Summarization Corpus](https://arxiv.org/abs/2004.14900)|
| 31 | Dialogue(Debate)|Argumentative Dialogue Summary Corpus |[Using Summarization to Discover Argument Facets in Online Idealogical Dialog](https://www.aclweb.org/anthology/N15-1046.pdf)|NAACL15|
|32|WCEP|多文档新闻领域|[A Large-Scale Multi-Document Summarization Dataset from the Wikipedia Current Events Portal](https://arxiv.org/abs/2005.10070)|ACL20 short|
|33|ArgKP|argument-to-key point mapping|[From Arguments to Key Points: Towards Automatic Argument Summarization](https://arxiv.org/abs/2005.01619)|ACL20|
|34|CRD3|对话|Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset|ACL20|
|35|Gazeta|俄语新闻|[Dataset for Automatic Summarization of Russian News](https://arxiv.org/pdf/2006.11063.pdf)||
|36|MIND|英文新闻推荐，标题生成，摘要，类别分类，实体信息|[MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)|ACL20|
|37|public_meetings|法语会议（只是测试集）|[Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation](Align then Summarize: Automatic Alignment Methods for Summarization Corpus Creation)|LREC|
|38|Enron|Email|[Building a Dataset for Summarization and Keyword Extraction from Emails](http://www.lrec-conf.org/proceedings/lrec2014/pdf/1037_Paper.pdf)|2014| 349 emails and threads|
|39|Columbia|Email|[Summarizing Email Threads](https://www.aclweb.org/anthology/N04-4027.pdf)|2004|96 email threads,average of 3.25 email|
|40|BC3|Email||[A publicly available annotated corpus for supervised email summarization.](https://www.ufv.ca/media/assets/computer-information-systems/gabriel-murray/publications/aaai08.pdf)|40 email threads (3222 sentences)|


## Multi-modal
| Paper | Conference |Highlights|
| :---: | :---: | :---: |
|[Text-Image-Video Summary Generation Using Joint Integer Linear Programming](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_24)|ECIR2020|generating an extractive multimodal output containing text, images and videos from a multi-modal input|
|[Aspect-Aware Multimodal Summarization for Chinese E-Commerce Products](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LiH.902.pdf)|AAAI20|
|[Convolutional Hierarchical Attention Network for Query-Focused Video Summarization](https://arxiv.org/abs/2002.03740)|AAAI20|
|[Multimodal Summarization with Guidance of Multimodal Reference](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhuJ.1133.pdf)|AAAI20|
|[EmotionCues: Emotion-Oriented Visual Summarization of Classroom Videos](https://ieeexplore.ieee.org/document/8948010)|IEEE2020|
|[A Survey on Automatic Summarization Using Multi-Modal Summarization System for Asynchronous Collections](http://www.ijirset.com/upload/2019/february/4_shilpa_IEEE.pdf)||
|[Extractive summarization of documents with images based on multi-modal RNN](https://research.aston.ac.uk/en/publications/extractive-summarization-of-documents-with-images-based-on-multi-)||
|[Keep Meeting Summaries on Topic: Abstractive Multi-Modal Meeting Summarization](https://www.aclweb.org/anthology/P19-1210/)|ACL19|
|[Multimodal Abstractive Summarization for How2 Videos](https://www.aclweb.org/anthology/P19-1659/) | ACL19 |
|[MSMO: Multimodal Summarization with Multimodal Output](https://www.aclweb.org/anthology/D18-1448/)|EMNLP18|
|[Abstractive Text-Image Summarization Using Multi-Modal Attentional Hierarchical RNN](https://www.aclweb.org/anthology/D18-1438/)|EMNLP18|
|[Multi-modal Sentence Summarization with Modality Attention and Image Filtering](https://www.ijcai.org/Proceedings/2018/0577.pdf) | IJCAI18 |
|[How2: A Large-scale Dataset for Multimodal Language Understanding](https://arxiv.org/abs/1811.00347)|NIPS18|
|[Multimodal Abstractive Summarization for Open-Domain Videos](https://nips2018vigil.github.io/static/papers/accepted/8.pdf) | NIPS18|
|[Read, Watch, Listen, and Summarize: Multi-Modal Summarization for Asynchronous Text, Image, Audio and Video](https://ieeexplore.ieee.org/document/8387512)|IEEE18|
|[Fusing Verbal and Nonverbal Information for Extractive Meeting Summarization](https://dl.acm.org/doi/10.1145/3279981.3279987)|GIFT18|
|[Multi-modal Summarization for Asynchronous Collection of Text, Image, Audio and Video](https://www.aclweb.org/anthology/D17-1114/) | EMNLP17 |
|[Meeting Extracts for Discussion Summarization Based on Multimodal Nonverbal Information](https://dl.acm.org/doi/10.1145/2993148.2993160)|ICMI16|
|[Summarizing a multimodal set of documents in a Smart Room](https://www.aclweb.org/anthology/L12-1524/)|LREC2012|
|[Multi-modal summarization of key events and top players in sports tournament videos](https://eprints.qut.edu.au/43479/1/WACV_266_%281%29.pdf)|2011 IEEE Workshop on Applications of Computer Vision|
|[Multimodal Summarization of Complex Sentences](https://www.cs.cmu.edu/~jbigham/pubs/pdfs/2011/multimodal_summarization.pdf)||
|[Summarization of Multimodal Information](http://www.lrec-conf.org/proceedings/lrec2004/pdf/502.pdf)|LREC2004|
| [Multimodal Summarization of Meeting Recordings](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.862.6509&rep=rep1&type=pdf)|ICME 2003|


## Concept-map-based
| Paper | Conference |
| :---: | :---: |
| Fast Concept Mention Grouping for Concept Map–based Multi-Document Summarization|NAACL19|
| Bringing Structure into Summaries : Crowdsourcing a Benchmark Corpus of Concept Maps | EMNLP17 |

## Timeline
| Paper | Conference |
| :---: | :---: |
|[Examining the State-of-the-Art in News Timeline Summarization](https://arxiv.org/abs/2005.10107)|ACL20|
|Learning towards Abstractive Timeline Summarization|IJCAI19|


## Opinion
| Paper | Conference |
| :---: | :---: |
|[Unsupervised Opinion Summarization as Copycat-Review Generation](https://arxiv.org/abs/1911.02247)|ACL20|
|[Unsupervised Opinion Summarization with Noising and Denoising](https://arxiv.org/abs/2004.10150)|ACL20|
|[OPINIONDIGEST: A Simple Framework for Opinion Summarization](https://arxiv.org/abs/2005.01901)|ACL20|
|[Weakly-Supervised Opinion Summarization by Leveraging External Information](https://arxiv.org/abs/1911.09844)|AAAI20|
| [MeanSum : A Neural Model for Unsupervised Multi-Document Abstractive Summarization](https://arxiv.org/abs/1810.05739)|ICML19


## Reinforcement Learning
| Paper | Conference |
| :---: | :---: |
|Answers Unite! Unsupervised Metrics for Reinforced Summarization Models|EMNLP19|
|Deep Reinforcement Learning with Distributional Semantic Rewards for Abstractive Summarization|EMNLP19|
| Reinforced Extractive Summarization with Question-Focused Rewards|ACL18|
| Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting|ACL18|
| Multi-Reward Reinforced Summarization with Saliency and Entailment|NAACL18|
| Ranking Sentences for Extractive Summarization with Reinforcement Learning|NAACL18|
| A Deep Reinforced Model For Abstractive Summarization|ICLR18|

## Reward Learning
| Paper | Conference |
| :---: | :---: |
|[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)||
|[Better Rewards Yield Better Summaries: Learning to Summarise Without References](https://arxiv.org/abs/1909.01214)|EMNLP19|

## Sentence Summarization
| Paper | Conference |
| :---: | :---: |
|[Discrete Optimization for Unsupervised Sentence Summarization with Word-Level Extraction](https://arxiv.org/abs/2005.01791)|ACL20|
|[Keywords-Guided Abstractive Sentence Summarization](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiH.1493.pdf)|AAAI20|

## Evaluation
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
|[SummEval: Re-evaluating Summarization Evaluation](https://arxiv.org/abs/2007.12626)|||
|[Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228)|ACL20|* Question generation(BART, NewsQA), * Question Answering(BERT, SQuAD2.0), * Answer similarity(Token level F1)|


## Controlled
| Paper | Conference | Highlights |
| :---: | :---: | :---: |
| [Length-controllable Abstractive Summarization by Guiding with Summary Prototype](https://arxiv.org/abs/2001.07331)||
|[Positional Encoding to Control Output Sequence Length](https://www.aclweb.org/anthology/N19-1401/)|NAACL19||
|[Query Focused Abstractive Summarization: Incorporating Query Relevance, Multi-Document Coverage, and Summary Length Constraints into seq2seq Models](https://arxiv.org/abs/1801.07704)||
| [Controllable Abstractive Summarization](https://arxiv.org/abs/1711.05217)|ACL2018 Workshop|
|[Controlling Length in Abstractive Summarization Using a Convolutional Neural Network](https://www.aclweb.org/anthology/D18-1444/)|EMNLP18||
|[Controlling Output Length in Neural Encoder-Decoders](https://www.aclweb.org/anthology/D16-1140/)|EMNLP16||

## Others
| Paper | Conference |
| :---: | :---: |
|[Abstractive Summarization of Spoken and Written Instructions with BERT](https://arxiv.org/abs/2008.09676)||
|[A Discourse-Aware Neural Extractive Model for Text Summarization](http://www.cs.utexas.edu/~jcxu/material/ACL20/DiscoBERT_ACL2020.pdf)|ACL20|
|[StructSum: Incorporating Latent and Explicit Sentence Dependencies for Single Document Summarization](https://arxiv.org/abs/2003.00576)||
|Leveraging Code Generation to Improve Code Retrieval and Summarization via Dual Learning|WWW20|
|[AutoSurvey: Automatic Survey Generation based on a Research Draft](https://www.ijcai.org/Proceedings/2020/0761.pdf)|IJCAI20|
|[Neural Abstractive Summarization with Structural Attention](https://arxiv.org/abs/2004.09739)|IJCAI20|
|[A Unified Model for Financial Event Classification, Detection and Summarization](https://www.ijcai.org/Proceedings/2020/644)|IJCAI20|
|[Extractive Summarization as Text Matching](http://pfliu.com/paper/ACL2020_MatchingSum.pdf)|ACL20|good analysis|
|[Discriminative Adversarial Search for Abstractive Summarization](https://arxiv.org/abs/2002.10375)|ICML20|
|[Joint Parsing and Generation for Abstractive Summarization](https://arxiv.org/abs/1911.10389)|AAAI20|
|Controlling the Amount of Verbatim Copying in Abstractive Summarization|AAAI20|
|[GRET：Global Representation Enhanced Transformer](https://arxiv.org/abs/2002.10101)|AAAI20 |
|Improving Abstractive Text Summarization with History Aggregation||
|Co-opNet: Cooperative Generator–Discriminator Networks for Abstractive Summarization with Narrative Flow||
|Contrastive Attention Mechanism for Abstractive Sentence Summarization|EMNLP19|
|Extractive Summarization of Long Documents by Combining Global and Local Context|EMNLP19|
|Countering the Effects of Lead Bias in News Summarization via Multi-Stage Training and Auxiliary Losses|EMNLP19|
|An Entity-Driven Framework for Abstractive Summarization|EMNLP19|
|Concept Pointer Network for Abstractive Summarization|EMNLP19|
| Countering the Effects of Lead Bias in News Summarization via Multi-Stage Training and Auxiliary Losses | EMNLP19 |
| Neural Extractive Text Summarization with Syntactic Compression | EMNLP19 |
|Neural Extractive Text Summarization with Syntactic Compression|EMNLP19|
|Reading Like HER: Human Reading Inspired Extractive Summarization|EMNLP19|
|BottleSum: Unsupervised and Self-supervised Sentence Summarization using the Information Bottleneck Principle|EMNLP19|
|Abstract Text Summarization: A Low Resource Challenge|EMNLP19|
|Attention Optimization for Abstractive Document Summarization|EMNLP19|
|Mem2Mem-Learning to Summarize Long Texts with Memory-to-Memory Transfer|ICLR20 under review|
| Inducing Document Structure for Aspect-based Summarization|ACL19|
| Summary Refinement through Denoising | RANLP 2019|
|Generating Summaries with Topic Templates and Structured Convolutional Decoders|ACL19|
|Sentence Centrality Revisited for Unsupervised Summarization|ACL19|
|BiSET: Bi-directional Selective Encoding with Template for Abstractive Summarization|ACL19|
| HIGHRES: Highlight-based Reference-less Evaluation of Summarization | ACL19 |
| Searching for Effective Neural Extractive Summarization: What Works and What's Next | ACL19 |
| Scoring Sentence Singletons and Pairs for Abstractive Summarization|ACL19|
| Guiding Extractive Summarization with Question-Answering Rewards|NAACL19|
|Single Document Summarization as Tree Induction|NAACL19|
|LeafNATS: An Open-Source Toolkit and Live Demo System for Neural Abstractive Text Summarization|NAACL19|
| Jointly Extracting and Compressing Documents with Summary State Representations|NAACL19|
| Structured Neural Summarization|ICLR19|
| DeepChannel: Salience Estimation by Contrastive Learning for Extractive Document Summarization|AAAI19|
| Retrieve, Rerank and Rewrite: Soft Template Based Neural Summarization|ACL18|
| Extractive Summarization with SWAP-NET: Sentences and Words from Alternating Pointer Networks|ACL18|
| Neural Latent Extractive Document Summarization|ACL18|
| Soft Layer-Specific Multi-Task Summarization with Entailment and Question Generation|ACL18|
| A Unified Model for Extractive and Abstractive Summarization using Inconsistency Loss|ACL18|
| Neural Document Summarization by Jointly Learning to Score and Select Sentences|ACL18|
|Abstractive Document Summarization via Bidirectional Decoder|ADMA18|
| Entity Commonsense Representation for Neural Abstractive Summarization | NAACL18|
| Entity Commonsense Representation for Neural Abstractive Summarization | NAACL18 |
| A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents | NAACL18 |
| Relational Summarization for Corpus Analysis | NAACL18 |
| Deep Communicating Agents for Abstractive Summarization | NAACL18 |
| Guiding Generation for Abstractive Text Summarization based on Key Information Guide Network|NAACL18|
| A Semantic QA-Based Approach for Text Summarization Evaluation|AAAI18|
| Generative Adversarial Network for Abstractive Text Summarization|AAAI18|
| Content Selection in Deep Learning Models of Summarization|EMNLP18|
| Improving Neural Abstractive Document Summarization with Explicit Information Selection Modeling|EMNLP18|
| Improving Neural Abstractive Document Summarization with Structural Regularization|EMNLP18|
| Closed-Book Training to Improve Summarization Encoder Memory|EMNLP18|
| Bottom-Up Abstractive Summarization|EMNLP18|
| Get To The Point: Summarization with Pointer-Generator Networks|ACL17|
|Selective Encoding for Abstractive Sentence Summarization|ACL17|
| Abstractive Document Summarization with a Graph-Based Attentional Neural Model|ACL17|
| Extractive Summarization Using Multi-Task Learning with Document Classification|EMNLP17| 
| Deep Recurrent Generative Decoder for Abstractive Text Summarization | EMNL17 |
| SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents|AAAI17|
| Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond|CoNLL16|
| A Neural Attention Model for Abstractive Sentence Summarization|EMNLP15|
| Toward Abstractive Summarization Using Semantic Representations|NAACL15|
| Abstractive Meeting Summarization with Entailment and Fusion||




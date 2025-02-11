# Knowledge Graph Attention Network
This is PyTorch implementation for the paper:
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

If you want to use codes and datasets in your research, please contact the paper authors and cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirement
The code has been tested running under Python 3.7.10. The required packages are as follows:
* torch == 1.6.0
* numpy == 1.21.4
* pandas == 1.3.5
* scipy == 1.5.2
* tqdm == 4.62.3
* scikit-learn == 1.0.1

## Run the Codes
* FM
```
python main_nfm.py --model_type fm --data_name amazon-book
```
* NFM
```
python main_nfm.py --model_type nfm --data_name amazon-book
```
* BPRMF
```
python main_bprmf.py --data_name amazon-book
```
* ECFKG
```
python main_ecfkg.py --data_name amazon-book
```
* CKE
```
python main_cke.py --data_name amazon-book
```
* KGAT
```
python main_kgat.py --data_name amazon-book
```

## Results
With my code, following are the results of each model when training with dataset `amazon-book`.

| Model                                             | Best Epoch | Precision@20 | Recall@20 | NDCG@20 |
| :---:                                             | :---:      | :---:        | :---:     | :---:   |
| FM                                                | 370        | 0.0154       | 0.1478    | 0.0784  |
| NFM                                               | 140        | 0.0137       | 0.1309    | 0.0696  |
| BPRMF                                             | 330        | 0.0146       | 0.1395    | 0.0736  |
| ECFKG                                             |  10        | 0.0134       | 0.1264    | 0.0663  |
| CKE                                               | 320        | 0.0145       | 0.1394    | 0.0733  |
| KGAT <br> (agg: bi-interaction; lap: random-walk) | 280        | 0.0150       | 0.1440    | 0.0766  |
| KGAT <br> (agg: bi-interaction; lap: symmetric)   | 200        | 0.0149       | 0.1428    | 0.0755  |
| KGAT <br> (agg: graphsage;      lap: random-walk) | 450        | 0.0147       | 0.1430    | 0.0747  |
| KGAT <br> (agg: graphsage;      lap: symmetric)   | 160        | 0.0146       | 0.1410    | 0.0735  |
| KGAT <br> (agg: gcn;            lap: random-walk) | 280        | 0.0149       | 0.1440    | 0.0760  |
| KGAT <br> (agg: gcn;            lap: symmetric)   | 670        | 0.0150       | 0.1448    | 0.0768  |

## Related Papers
* FM
    * Proposed in [Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002), SIGIR 2011.

* NFM
    * Proposed in [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777), SIGIR 2017.

* BPRMF
    * Proposed in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167), UAI 2009.
    * Key point: 
        * Replace point-wise with pair-wise.

* ECFKG
    * Proposed in [Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation](https://arxiv.org/abs/1805.03352), Algorithms 2018.
    * Implementation by the paper authors: [https://github.com/evison/KBE4ExplainableRecommendation](https://github.com/evison/KBE4ExplainableRecommendation)
    * Key point: 
        * Introduce Knowledge Graph to Collaborative Filtering

* CKE
    * Proposed in [Collaborative Knowledge Base Embedding for Recommender Systems](https://dl.acm.org/citation.cfm?id=2939673), KDD 2016.
    * Key point: 
        * Leveraging structural content, textual content and visual content from the knowledge base.
        * Use TransR which is an approach for heterogeneous network, to represent entities and relations in distinct semantic space bridged by relation-specific matrices.
        * Performing knowledge base embedding and collaborative filtering jointly.

* KGAT
    * Proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD 2019.
    * Implementation by the paper authors: [https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * Key point:
        * Model the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.
        * Train KG part and CF part in turns.
        


# Knowledge Graph Attention Network
This is our Tensorflow implementation for the paper:

>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). [KGAT: Knowledge Graph Attention Network for Recommendation](xx). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.

## Citation 
If you want to use our codes in your research, please cite:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat-Seng Chua},
  title     = {KGAT: Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  year      = {2019}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.12.0
* numpy == 1.15.4
* scipy == 1.1.0
* sklearn == 0.20.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in Model/utility/parser.py).
* Yelp2018 dataset
```
python Main.py --model_type kgat --alg_type bi --dataset gowalla --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 400 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
```

* Amazon-book dataset
```
python Main.py --model_type kgat --alg_type bi --dataset amazon-book --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 400 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
```


* Last-fm dataset
```
python Main.py --model_type kgat --alg_type bi --dataset last-fm --regs [1e-5,1e-5] --layer_size [64,32,16] --embed_size 64 --lr 0.0001 --epoch 400 --verbose 1 --save_flag 1 --pretrain -1 --batch_size 1024 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1] --use_att True --use_kge True
```

Some important arguments:
* `model_type`
  * It specifies the type of model.
  * Here we provide six options, including KGAT and five baseline models:
    * `kgat` (by default), proposed in [KGAT: Knowledge Graph Attention Network for Recommendation](xx), KDD2019. Usage: `--model_type kgat`.
    * `bprmf`, proposed in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=1795167), UAI2009. Such model only uses user-item interactions. Usage: `--model_type bprmf`.
    * `fm`, proposed in [Fast context-aware recommendations with factorization machines](https://dl.acm.org/citation.cfm?id=2010002), SIGIR2011. Usage: `--model_type fm`.
    * `nfm`, proposed in [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777), SIGIR2017. Usage: `--model_type nfm`.
    * `cke`, proposed in [Collaborative Knowledge Base Embedding for Recommender Systems](https://dl.acm.org/citation.cfm?id=2939673), KDD2016. Usage: `--model_type cke`.
    * `cfkg`, proposed in [Learning Heterogeneous Knowledge Base Embeddings for Explainable Recommendation](https://arxiv.org/abs/1805.03352), Algorithm2018. Usage: `--model_type cfkg`.
  * You can find other baselines, such as RippleNet, MCRec, and GC-MC, in Github.

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
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Example to Run the Codes

# Training Log of KGAT

To demonstrate the **reproducibility of our best performance** and **faciliate the researchers to track their own trainings**, here I offer the training log for KGAT on three datasets, recording the changes of `loss` and four evaluation metrics: `recall@k`, `precision@k`, `hit@k`, and `ndcg@k`, every 10 epochs.

## Record Example
One record example is shown as follows:
```
Epoch 19 [238.5s + 104.4s]: train==[30.47060=16.64901 + 13.32211 + 0.49948], recall=[0.06590, 0.19621], precision=[0.01598, 0.00987], hit=[0.25201, 0.55005], ndcg=[0.08068, 0.15693]
save the weights in path:  weights/yelp2018/kgat_si_sum_bi_l3/64-32-16/l0.0001_r1e-05-1e-05
```
where:
* `[238.5s + 104.4s]` shows the time cost for one training and one testing, respectively;
* `train==[30.47060=16.64901 + 13.32211 + 0.49948]` records the loss of BPR loss for recommendation, BPR loss for knowledge graph embedding, and L2 regularization terms for both previous phases;
* `recall=[0.06590, 0.19621]` illustrates the `recall@20` and `recall@100` scores; analogously for precision, hit, and ndcg scores.

The final perfroamnce is shown as:
```
Best Iter=[95]@[166677.6]	recall=[0.08820	0.12068	0.14174	0.15773	0.17082], precision=[0.03521	0.02661	0.02217	0.01939	0.01741], hit=[0.36111	0.46397	0.52597	0.56730	0.59930], ndcg=[0.13660	0.16881	0.19020	0.20689	0.22085]
```
where:
* `Best Iter=[95]@[166677.6]` shows that the best performance appears at the `95*10` epoch (since we show the training log every 10 epochs);
* `recall=[0.08820	0.12068	0.14174	0.15773	0.17082]` displays the final performance w.r.t. `recall@k`, where `k` spans from `20`, `40`, `60`, `80` to `100`.

## Some Points
Here I would like to clarify some points:
* The training and testing time costs might be different based on the running machines.
* The training loss might be slightly different due to different random seeds.
* Please note that, while setting the hyperparameters `--Ks` as `[20,40,60,80,100]`, here we only show the results with k of `20` and `100`, due to the limited space; when the training is finished, you will obtain the scores with all setting `Ks`.


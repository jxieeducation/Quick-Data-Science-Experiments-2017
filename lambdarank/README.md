This is the LightGBM example.
https://github.com/Microsoft/LightGBM/blob/master/examples/lambdarank/train.conf

Here is the output: The training NDCG should increase on the training data...
[LightGBM] [Info] Iteration:99, training ndcg@1 : 0.977162
[LightGBM] [Info] Iteration:99, training ndcg@3 : 0.988908
[LightGBM] [Info] Iteration:99, training ndcg@5 : 0.983929
[LightGBM] [Info] Iteration:99, valid_1 ndcg@1 : 0.603619
[LightGBM] [Info] Iteration:99, valid_1 ndcg@3 : 0.633659
[LightGBM] [Info] Iteration:99, valid_1 ndcg@5 : 0.668626
[LightGBM] [Info] 92.676328 seconds elapsed, finished iteration 99
[LightGBM] [Info] Iteration:100, training ndcg@1 : 0.977162
[LightGBM] [Info] Iteration:100, training ndcg@3 : 0.988908
[LightGBM] [Info] Iteration:100, training ndcg@5 : 0.984302
[LightGBM] [Info] Iteration:100, valid_1 ndcg@1 : 0.603619
[LightGBM] [Info] Iteration:100, valid_1 ndcg@3 : 0.633308
[LightGBM] [Info] Iteration:100, valid_1 ndcg@5 : 0.670439
[LightGBM] [Info] 94.130038 seconds elapsed, finished iteration 100

LambdaRank paper: http://research.microsoft.com/en-us/um/people/cburges/papers/LambdaRank.pdf

NDCG:
https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
(0 to 1), 1 is ideal
K or L is the # of recommended entities
relavence --> smaller the better
https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain

Target function - e.g. accuracy / error rate - nonsmooth, many nondifferentiables
Optimization function - e.g. MSE, LogLoss - smooth

Target-optimization mismatch is dangerous


**EDIT: I got lost... need to read ranknet I guess first...
**see other page in the dir.

# A3C

### Details
Trains around a day for Pong.
- https://gym.openai.com/evaluations/eval_YynBVD1QH22Qx8JAfpBqw#reproducibility

### Run

```sh
$ python3.5 train.py --model_dir ./tmp --env Pong-v0 --t_max 20 --eval_every 60 --parallelism 16
```

### Reference
- https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
- https://github.com/ikostrikov/pytorch-a3c

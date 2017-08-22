# A3C

### Details
Trains around a half day for Pong and a day for Breakout.
- https://gym.openai.com/evaluations/eval_YynBVD1QH22Qx8JAfpBqw#reproducibility
- https://gym.openai.com/evaluations/eval_vkO26nnMRCeiU1tYRokSTQ#reproducibility

### Run

```sh
$ python3.5 train.py --model_dir ./tmp --env Pong-v0 --t_max 20 --eval_every 60 --parallelism 16
```

### Submit
```sh
$ python3.5 submit.py --ckpt_dir ./models/checkpoints-breakout --env Breakout-v0
```

### Reference
- https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
- https://github.com/ikostrikov/pytorch-a3c

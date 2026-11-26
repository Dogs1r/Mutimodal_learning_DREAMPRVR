# AdaptivePRVR 固定六槽实验

本地基线分支为 `baseline/charades-reproduced`（4764b19），开发分支为 `adaptiveprvr/fixed-six`。下面的训练从随机初始化开始，所有实验共用固定的六个槽和十步扩散。

| --experiment | 时间窗 | 扩散目标 |
|---|---|---|
| baseline | 原 DreamPRVR | 同视频描述均值扰动 |
| fixed-mean | 固定中心与宽度 | 均值扰动 |
| fixed-sentence | 固定中心与宽度 | 按句分配至局部槽 |
| learned-mean | 视频预测中心与宽度 | 均值扰动 |
| learned-sentence | 视频预测中心与宽度 | 按句分配至局部槽 |

事件版本均将窗口内汇聚的视觉向量注入去噪条件，并加入逐句的局部检索损失。逐句版本将每句话分配给两个最相似的槽，同一槽的句向量按分配质量平均；未获分配的槽仍参与视频编码。建库仅使用视频，主排序沿用上游片段与帧分数。

## 运行

```bash
export BASE=/home/dogsir/multimodal_learning/dreamprvr-work
cd "$BASE/code/src"
PY="$BASE/conda/envs/dreamprvr/bin/python"
"$PY" -B main.py -d cha --gpu 0 --experiment learned-sentence --run-name learned-sentence-s0 --seed 0 --batch-size 64 --epochs 100
"$PY" -B main.py -d cha --gpu 0 --experiment learned-sentence --run-name learned-sentence-s0 --eval-dev --resume "$BASE/runs/AdaptivePRVR/learned-sentence-s0/best.ckpt"
PYTHONPATH=. "$PY" -B -m unittest discover -s ../tests
```

每个新训练使用新的 run-name。运行目录记录配置、训练和开发描述清单、日志和 best.ckpt。开发集使用原训练集中的 534 个视频（1,260 句），训练集为 4,804 个视频（11,148 句）。选模依次比较开发 R@1 和 SumR。恢复训练时带上原先的实验名、种子、批量及总轮数，指定已有权重。官方网站测试划分使用 `--eval`，独立开发划分使用 `--eval-dev`。省略 `--experiment` 时走上游配置和评测口径。

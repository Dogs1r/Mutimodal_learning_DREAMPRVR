# AdaptivePRVR 原生 CLIP 查询与局部对齐

## Charades-STA 实验

复用种子 0 的 DreamPRVR 主干，开发阶段拟合 4,804 个训练视频、11,148 条查询，在独立的 534 个视频、1,260 条查询上选模。官方测试阶段重新用 5,338 个训练视频、12,408 条描述拟合映射，在 1,334 个候选视频、3,720 条查询上检索。主干和 CLIP 参数冻结。测试查询仅以句子作为文本编码器输入。候选视频只读预提取的视频特征。以视频 ID 核对训练、评测互斥。

原生 CLIP ViT-B/32 文本编码器从原始句子提取 512 维向量。在全部训练描述上，重编码结果与数据包特征的余弦相似度均值 0.9999992，最低 0.9999967。每条训练描述在其视频的 16 个时间窗口上做相似度加权池化，softmax 温度为 1/60，以此拟合从 CLIP 文本到局部视觉向量、从局部视觉向量到 CLIP 文本的两组岭回归。各方向对候选视频取最高三个窗口的均值，再按查询在全库标准化。

全库先用上轮双向对齐分加新分支的分数排序。在前 5 名内，保留新分支，将上轮双向分替换为上轮单向分。开发集比较直接使用 CLIP、使用新训练的双向映射，融合权重 0.125、0.25、0.5、1、2，以及是否精排。按 R@1、SumR 依次选模，锁定训练对齐、0.125、精排。开发集上轮双向 7.46 / 148.02，本轮 7.62 / 148.89（R@1 / SumR）。

| 官方测试 | R@1 | R@5 | R@10 | R@100 | SumR |
|---|---:|---:|---:|---:|---:|
| 本地同主干 DreamPRVR | 2.31 | 8.98 | 14.25 | 52.98 | 78.52 |
| 上轮双向对齐 | 2.42 | 9.06 | 14.92 | 53.63 | 80.03 |
| 上轮双向对齐，仅加前五单向精排 | 2.58 | 9.06 | 14.92 | 53.63 | 80.19 |
| 原生 CLIP 单独打分 | 1.05 | 4.22 | 6.64 | 30.62 | 42.53 |
| 本地 DreamPRVR + 原生 CLIP 直接融合 | 2.61 | 9.35 | 14.87 | 53.60 | 80.43 |
| 上轮双向 + 原生 CLIP 直接融合 | 2.50 | 9.46 | 15.46 | 54.03 | 81.45 |
| 加训练对齐，关闭精排 | 2.72 | 9.11 | 15.03 | 53.68 | 80.54 |
| 开发集锁定版本，训练对齐加精排 | 2.80 | 9.11 | 15.03 | 53.68 | 80.62 |

对照上轮双向，首位新增 23 条、移出 9 条。按真值视频聚类配对 bootstrap（10,000 次，种子 42），R@1 差的 95% 区间为 [+0.08, +0.71] 个百分点，SumR 差为 [−0.38, +1.55] 分。对照同主干原版，两项区间分别为 [+0.03, +0.96]、[+0.58, +3.61]。结果基于一个主干种子，Charades 测试集已用于此前多轮研究。上轮单向映射在本轮分块实现上 R@100 有一条查询差异，基线与上轮双向的各项召回均复现。作者发布的原方法权重此前实测 SumR 80.0，输入为 I3D 和 RoBERTa，本轮额外使用 CLIP 视觉和文本编码。

## 换机和换数据集

代码可以在单张 CUDA GPU 上运行。作者在 A100 40 GB 上训练了 TVR 和 ActivityNet Captions 原版。此分支为二者的原版配置加入 BASE 路径覆盖，未设置时保留上游行为。本机以两个数据集各自的输入维度，分别实测了一次合成批次的前向、损失、反向和参数更新。在这两个真实数据集上的训练、显存峰值和新增检索效果待目标机实测。旧的 main.py --experiment 槽监督实验仍限定 Charades。新增检索分支使用 native_clip.py。

1. 在目标机器准备兼容驱动和 PyTorch。Charades 实测 PyTorch 2.7.1+cu128、Python 3.11。其他 Python 依赖包括 numpy、h5py、tqdm、ipdb、easydict、pyyaml、seaborn、matplotlib。查询编码还需 tokenizers 和 OpenAI ViT-B/32 TorchScript 权重及对应 tokenizer.json。无需搬运本机 Conda 环境。
2. 设置 BASE 为服务器上的工作目录，数据放在 $BASE/data/tvr 或 $BASE/data/activitynet，训练输出位于 $BASE/runs。原版需要 TextData 中的训练/验证/测试描述和 RoBERTa 特征，FeatureData 中的 feature.hdf5、video2frames.txt；TVR 使用 i3d_resnet，ActivityNet 使用 i3d。按上游准备数据集的其他随附文件。
3. 使用对应数据集的 DreamPRVR 权重和训练/评测清单。新增分支还需要与文本编码器匹配的逐时间步 CLIP 视觉特征 HDF5，每个视频键对应非空 [T,512] 数组。文本 HDF5 的键为查询 ID，内容 [1,512] 或 [512]，可从原始描述生成。所有视觉特征需要使用同一套视频 ID，清单的查询 ID 以视频 ID 加 # 开头。更换数据集时重新训练映射并在独立开发集选择权重。候选集由评测清单包含的视频组成。大库按视频分块读取窗口，最终查询×视频分数矩阵仍占用显存。多卡和其他数据集的真实全量显存需求未验证。

原版训练示例，在代码的 src 下运行；ActivityNet 用 act 代替 tvr。

```bash
export BASE=/path/to/work
python main.py -d tvr --gpu 0
```

新增分支示例。先设置 CKPT、TRAIN、DEV、TEST、VIS，分别为对应权重、互斥训练/开发/测试描述清单和 CLIP 视频 HDF5。WEIGHTS、TOKENIZER 是下面两项公开资源的本地路径；OUT 指向该数据集独立的输出目录。Charades 原版发布包中 val 与 test 重叠，本次 DEV 使用从官方训练集按视频留出的清单，具体本机输入记录在每阶段的 metadata.json。

```bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
python encode_clip_queries.py --weights "$WEIGHTS" --tokenizer "$TOKENIZER" \
  --output "$OUT/queries.h5" "$TRAIN" "$DEV" "$TEST"
common=(--data-root "$BASE/data" --visual "$VIS" --train-text "$OUT/queries.h5" --query-text "$OUT/queries.h5")
python native_clip.py dev "${common[@]}" --checkpoint "$CKPT" --train "$TRAIN" \
  --eval "$DEV" --output "$OUT/dev"
python native_clip.py test "${common[@]}" --checkpoint "$CKPT" --train "$TRAIN" \
  --eval "$TEST" --selection "$OUT/dev/selection.json" --output "$OUT/test"
```

如有全训练集专用主干，在 test 命令替换 CKPT、TRAIN 为全量权重和训练清单，并确保训练文本特征包含全量描述。本机 Charades 使用数据包训练文本作为 --train-text，重编码的查询作为 --query-text。脚本输出 maps.pt、selection.json、trials.json、metrics.json、metadata.json 和逐查询 ranks.tsv。权重 SHA-256 为 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af。tokenizer.json SHA-256 为 b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842。[CLIP 权重](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) · [分词器](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json)

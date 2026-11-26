# AdaptivePRVR

基于 CVPR 2026 DreamPRVR 的两阶段视频检索实现。第一阶段在全库融合 DreamPRVR、双向局部对齐和 CLIP 文本匹配分数，第二阶段用训练得到的局部视觉对齐分数重排前五。源码源自上游提交 `34ae21e` 及本地实验提交 `13c5cd9`。[原作、作者和引用](UPSTREAM.md) · [实验报告](experiments/two-stage-top5-v1.md)

| Charades-STA，3,720 条查询 | R@1 | R@5 | R@10 | R@100 | SumR |
|---|---:|---:|---:|---:|---:|
| 同主干基线 | 2.31 | 8.98 | 14.25 | 52.98 | 78.52 |
| 两阶段检索 | 2.82 | 9.46 | 15.46 | 54.03 | 81.77 |

## 安装与验证

以下命令适用于单卡 Linux 环境。使用 Python 3.11、支持 CUDA 12.8 的 NVIDIA 驱动和 PyTorch 2.7.1；A100 使用同一 CUDA 构建。依赖和数据保存在各自服务器上。

```bash
git clone https://github.com/Dogs1r/Mutimodal_learning_DREAMPRVR.git
cd Mutimodal_learning_DREAMPRVR
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.ones(1,device='cuda').item())"
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Charades 数据与主干

在服务器上用 `BASE` 指定工作目录；三种主干配置也可默认使用仓库中的 work/。数据包与作者的基线权重入口见上游说明。将数据解压到 `$BASE/data/charades`；如果只有 feature.bin，使用下方转换命令一次。训练和评测描述位于 TextData，原版训练入口还会读取验证描述。已有实验主干的权重可从原机器单独复制；复现表中结果需要使用该实验的全量训练权重。作者发布的主干或在新机器重训的主干也可作为输入，分数会随权重改变。[上游数据与权重](UPSTREAM.md)

```bash
export BASE=/path/to/work
# 若尚未生成主干的视觉 HDF5
python src/Utils/convert_hdf5.py "$BASE/data/charades/FeatureData/i3d_rgb_lgi"
# 可选，在此机器训练主干
python src/main.py -d cha --gpu 0
```

两阶段检索预期使用以下输入。训练映射只访问训练描述和对应的视频。仓库提供代码，数据包、模型和输出单独存放。

```text
$BASE/
  data/charades/
    FeatureData/i3d_rgb_lgi/{feature.hdf5,video2frames.txt,new_clip_vit_32_charades_vid_features.hdf5}
    TextData/{charadestrain.caption.txt,charadesval.caption.txt,charadestest.caption.txt}
    TextData/{roberta_charades_query_feat.hdf5,clip_ViT_B_32_charades_query_feat.hdf5}
  weights/ViT-B-32.pt
  weights/clip-tokenizer/tokenizer.json
```

## 一条命令拟合与评测

设置 CHECKPOINT 为对应数据集的 DreamPRVR 主干权重。脚本自动编码原始查询、从训练集拟合映射、全库检索并在官方测试集输出 metrics.json、ranks.tsv、top5.tsv 和 maps.pt。若已有匹配的查询编码，可设置 QUERY_TEXT 复用 HDF5 并跳过文本编码。用 OUTPUT 指定新的实验目录。

```bash
mkdir -p "$BASE/weights/clip-tokenizer"
curl -fL https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o "$BASE/weights/ViT-B-32.pt"
curl -fL https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json -o "$BASE/weights/clip-tokenizer/tokenizer.json"
export CHECKPOINT=/path/to/charades-backbone.ckpt
CUDA_VISIBLE_DEVICES=0 bash scripts/test_charades.sh
```

## TVR 和 ActivityNet Captions

将数据分别置于 `$BASE/data/tvr` 和 `$BASE/data/activitynet`。主干训练入口分别为 `python src/main.py -d tvr --gpu 0` 和 `python src/main.py -d act --gpu 0`。上游在 A100 40 GB 上训练这两个主干。本项目的实际全量检索实验使用 RTX 5080 和 Charades，TVR 和 ActivityNet 的真实全量迁移仍需测量。

复用两阶段分支时，准备相同 CLIP 编码器产生的逐视频时间序列视觉特征，以及对应数据集的两套文本特征。以下命令在环境变量指向目标数据集文件后运行。训练和评测清单的视频 ID 应互斥，CLIP 视频 HDF5 按视频 ID 存储 [T,512]，CLIP 文本 HDF5 按查询 ID 存储 [1,512] 或 [512]。

```bash
python src/native_clip.py test --two-stage --checkpoint "$CHECKPOINT" --data-root "$BASE/data" \
  --train "$TRAIN_CAPTIONS" --eval "$TEST_CAPTIONS" --visual "$CLIP_VIDEO" \
  --train-text "$CLIP_TRAIN_TEXT" --query-text "$CLIP_QUERY_TEXT" --output "$OUTPUT"
```

两阶段入口沿用 Charades 的 0.125 融合系数和五个候选。可先把评测清单切换为独立开发集，测量迁移效果。显存随查询数与候选库大小增长。其他实验设计与指标见 [实验目录](experiments/)。

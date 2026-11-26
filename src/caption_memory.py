"""Fit train-only caption/visual mappings and retrieve over the entire gallery."""
import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Datasets.data_provider import (VisDataSet4PRVR, collate_frame_val, TxtDataSet4PRVR,
                                    collate_text_val, read_video_ids)
from Models.builder import get_models
from Utils.utils import load_ckpt


def norm(x):
    return F.normalize(x, dim=-1)


def ridge(x, y, reg=1.0):
    x = norm(x)
    mean_x, mean_y = x.mean(0), y.mean(0)
    x = x - mean_x
    gram = x.T @ x + reg * torch.eye(x.shape[1], device=x.device)
    return mean_x, mean_y, torch.linalg.solve(gram, x.T @ (y - mean_y))


def predict(x, mapping):
    mean_x, mean_y, weight = mapping
    return norm((norm(x) - mean_x) @ weight + mean_y)


def visual_features(path, vids):
    with h5py.File(path) as f:
        vectors = []
        for vid in vids:
            x = torch.from_numpy(f[vid][:]).float()
            # Sixteen local windows sample the complete video, including short clips.
            x = F.adaptive_avg_pool1d(norm(x).T[None], 16)[0].T
            vectors.append(norm(x))
    return torch.stack(vectors).cuda()


def fit(train, visual_path, text_path):
    q = norm(train['query'].cuda())
    with h5py.File(text_path) as f:
        clip_text = norm(torch.from_numpy(np.stack([f[i][0] for i in train['ids']])).cuda().float())
    windows = visual_features(visual_path, train['vids'])[train['truth']]
    attention = (windows * clip_text[:, None]).sum(-1).mul(60).softmax(-1)
    support = norm((windows * attention[:, :, None]).sum(1))
    return dict(query_map=ridge(q, clip_text), video_map=ridge(support, q),
                support=support, captions=q, videos=train['vids'],
                checkpoint=train['checkpoint'])


@torch.no_grad()
def baseline_scores(cfg, model, data, q):
    root = Path(cfg['data_root']) / cfg['collection']
    vis = VisDataSet4PRVR(str(root/'FeatureData'/cfg['visual_feature']/'feature.hdf5'),
                          {}, cfg, video_ids=data['vids'])
    scores = []
    for clips, frames, mask, _, _ in DataLoader(vis, batch_size=64, collate_fn=collate_frame_val):
        frame, clip = model.encode_context(clips.cuda(), frames.cuda(), mask.cuda())
        chunks = []
        for batch in q.split(64):
            chunks.append(cfg['clip_scale_w'] * model.get_clip_scale_scores(batch, clip)
                          + cfg['frame_scale_w'] * model.get_clip_scale_scores(batch, frame))
        scores.append(torch.cat(chunks))
    return torch.cat(scores, 1)


def standardize(s):
    return (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True).clamp_min(1e-6)


def score(q, video):
    # Allow a short local match, while suppressing isolated spikes.
    return torch.cat([(batch @ video.flatten(0, 1).T).view(len(batch), len(video), -1)
                      .topk(3, dim=2).values.mean(2) for batch in q.split(64)])


def components(data, bank, visual_path, cfg, model):
    if set(data['vids']) & set(bank['videos']) or bank['checkpoint'] != data['checkpoint']:
        raise ValueError('Memory/gallery overlap or mismatched DreamPRVR encoder')
    q = data['query'].cuda()
    windows = visual_features(visual_path, data['vids'])
    direct = score(predict(q, bank['query_map']), windows)
    projected = score(norm(q), predict(windows, bank['video_map']))
    memory = []
    for batch in windows.split(32):
        similarities = batch @ bank['support'].T
        values, idx = similarities.topk(32)
        memory.append(norm(((values / .04).softmax(-1).unsqueeze(-1) * bank['captions'][idx]).sum(-2)))
    votes = score(norm(q), torch.cat(memory))
    base = baseline_scores(cfg, model, data, q)
    return tuple(map(standardize, (base, direct, projected, votes)))


def measure(s, truth):
    order = s.argsort(1, descending=True)
    rank = (order == truth[:, None]).int().argmax(1) + 1
    metrics = {f'R@{k}': (rank <= k).sum().item() * 100 / len(rank) for k in (1, 5, 10, 100)}
    metrics['SumR'] = sum(metrics.values())
    return rank, metrics


def encode_queries(root, captions, checkpoint, cfg, model):
    vids = read_video_ids(captions)
    dataset = TxtDataSet4PRVR(captions, str(root/'TextData'/f"roberta_{cfg['collection']}_query_feat.hdf5"), cfg)
    ids, vectors = [], []
    for tokens, mask, _, names in DataLoader(dataset, batch_size=50, collate_fn=collate_text_val):
        vectors.append(model.encode_query(tokens.cuda(), mask.cuda()).cpu())
        ids.extend(names)
    lookup = {v: i for i, v in enumerate(vids)}
    truth = torch.tensor([lookup[q.split('#')[0]] for q in ids])
    return dict(vids=vids, ids=ids, truth=truth, query=torch.cat(vectors), checkpoint=str(checkpoint))


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('dev', 'test', 'full'))
    parser.add_argument('--base', type=Path, default=os.environ.get('BASE'),
                        help='Workspace containing data/ and runs/ (defaults to $BASE)')
    args = parser.parse_args()
    if args.base is None:
        parser.error('Set BASE or supply --base')
    runs = args.base / 'runs' / 'AdaptivePRVR'
    cache = runs / 'top20-v1'
    output = runs / 'caption-memory-v1'
    output.mkdir(exist_ok=True)
    root = args.base / 'data' / 'charades'
    visual = root/'FeatureData/i3d_rgb_lgi/new_clip_vit_32_charades_vid_features.hdf5'
    text = root/'TextData/clip_ViT_B_32_charades_query_feat.hdf5'
    if args.phase == 'dev':
        train = torch.load(cache/'train.pt', weights_only=True)
        bank = fit(train, visual, text)
        torch.save(bank, output/'bank.pt')
    elif args.phase == 'test':
        bank = torch.load(output/'bank.pt', weights_only=True)
    split = 'val' if args.phase == 'dev' else 'test'
    if args.phase == 'full':
        checkpoint = (runs / 'full-s0-b64-baseline/final.ckpt').resolve()
    else:
        data = torch.load(cache/(split+'.pt'), weights_only=True)
        checkpoint = data['checkpoint']
    cfg, weights, _, _, _ = load_ckpt(checkpoint)
    model = get_models(cfg).cuda().eval()
    model.load_state_dict(weights)
    if args.phase == 'full':
        train = encode_queries(root, root/'TextData/charadestrain.caption.txt', checkpoint, cfg, model)
        bank = fit(train, visual, text)
        torch.save(bank, output/'full-bank.pt')
        data = encode_queries(root, root/'TextData/charadestest.caption.txt', checkpoint, cfg, model)
    base, direct, projected, votes = components(data, bank, visual, cfg, model)
    truth = data['truth'].cuda()
    rank0, metrics0 = measure(base, truth)
    if 'rank' in data and not torch.equal(rank0.cpu(), data['rank']):
        mismatch = (rank0.cpu() != data['rank']).sum().item()
        if any(not torch.equal(rank0.cpu() <= k, data['rank'] <= k) for k in (1,5,10,100)):
            raise ValueError(f'Baseline recall changed ({mismatch} different ranks)')
        print('Baseline order differs on near ties:', mismatch)
    variants = {'clip': direct, 'aligned': (direct + projected)/2,
                'memory': (direct + projected + votes)/3}
    results = {'baseline': metrics0}
    ranks = {'baseline': rank0.tolist()}
    if args.phase == 'dev':
        selected = {}
        for name, extra in variants.items():
            scores = {a: measure(base + a * extra, truth)[1] for a in (.125,.25,.5,1.)}
            selected[name] = max(scores,key=lambda a:(scores[a]['R@1'],scores[a]['SumR'],-a))
        (output/'selection.json').write_text(json.dumps(selected,indent=2))
    else:
        selected = json.loads((output/'selection.json').read_text())
    for name, extra in variants.items():
        rank, metrics = measure(base + selected[name] * extra, truth)
        results[name] = metrics
        ranks[name] = rank.tolist()
    with (output/(args.phase+'-ranks.tsv')).open('w') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['query',*ranks])
        writer.writerows(zip(data['ids'],*ranks.values()))
    (output/(args.phase+'.json')).write_text(json.dumps(results,indent=2))
    print(selected,results)


if __name__ == '__main__':
    main()

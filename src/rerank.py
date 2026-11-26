"""Frozen DreamPRVR retrieval followed by six-window candidate reranking."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Datasets.data_provider import (TxtDataSet4PRVR, VisDataSet4PRVR, collate_frame_val,
                                    collate_text_val, read_video_ids)
from Models.builder import get_models
from Utils.utils import load_ckpt, set_seed


def windows(clips):
    # Six fixed intervals partition the full sequence of 32 I3D bins.
    means, differences = [], []
    for section in clips.tensor_split(6, dim=1):
        before, after = section.tensor_split(2, dim=1)
        means.append(section.mean(1))
        differences.append(after.mean(1) - before.mean(1))
    return torch.stack(means, 1), torch.stack(differences, 1)


@torch.no_grad()
def prepare(checkpoint, captions, output, bank_path=None):
    cfg, state, _, _, _ = load_ckpt(checkpoint)
    model = get_models(cfg).cuda().eval()
    model.load_state_dict(state)
    root = Path(cfg['data_root']) / cfg['collection']
    vids = read_video_ids(captions)
    vis = VisDataSet4PRVR(str(root/'FeatureData'/cfg['visual_feature']/'feature.hdf5'),
                          {}, cfg, video_ids=vids)
    txt = TxtDataSet4PRVR(captions, str(root/'TextData'/'roberta_charades_query_feat.hdf5'), cfg)
    vectors, descriptions = [], []
    for tokens, mask, _, ids in DataLoader(txt, batch_size=50, collate_fn=collate_text_val):
        vectors.append(model.encode_query(tokens.cuda(), mask.cuda()))
        descriptions.extend(ids)
    query = torch.cat(vectors)
    bank = None
    penalties = []
    if bank_path is not None:
        reference = torch.load(bank_path, weights_only=True)
        assert reference['checkpoint'] == str(checkpoint.resolve())
        assert set(reference['vids']).isdisjoint(vids)
        bank = reference['query'].cuda()
        del reference
    means, differences, blocks = [], [], []
    for clips, frames, mask, _, _ in DataLoader(vis, batch_size=64, collate_fn=collate_frame_val):
        avg, delta = windows(clips)
        means.append(avg)
        differences.append(delta)
        frames, clips = model.encode_context(clips.cuda(), frames.cuda(), mask.cuda())
        chunks = []
        for q in query.split(64):
            clip_score = model.get_clip_scale_scores(q, clips)
            frame_score = model.get_clip_scale_scores(q, frames)
            chunks.append(cfg['clip_scale_w'] * clip_score + cfg['frame_scale_w'] * frame_score)
        blocks.append(torch.cat(chunks).cpu())
        if bank is not None:
            # Estimate each gallery video's background affinity to training captions.
            bank_scores = []
            for q in bank.split(128):
                bank_scores.append(cfg['clip_scale_w'] * model.get_clip_scale_scores(q, clips)
                                   + cfg['frame_scale_w'] * model.get_clip_scale_scores(q, frames))
            bank_scores = torch.cat(bank_scores)
            penalties.append((0.03 * (torch.logsumexp(bank_scores / 0.03, dim=0)
                                      - math.log(len(bank_scores)))).cpu())
    scores = torch.cat(blocks, 1)
    lookup = {vid: i for i, vid in enumerate(vids)}
    truth = torch.tensor([lookup[q.split('#')[0]] for q in descriptions])
    ranks, candidates = [], []
    for rows, labels in zip(scores.split(64), truth.split(64)):
        order = (-rows.cuda()).argsort(dim=1)
        ranks.append((order == labels.cuda()[:, None]).int().argmax(1).cpu() + 1)
        candidates.append(order[:, :20].cpu())
    candidates = torch.cat(candidates)
    data = dict(query=query.cpu(), means=torch.cat(means), delta=torch.cat(differences),
                candidates=candidates, base=scores.gather(1, candidates), truth=truth,
                true_score=scores.gather(1, truth[:, None]).squeeze(1), rank=torch.cat(ranks),
                ids=descriptions, vids=vids, checkpoint=str(checkpoint.resolve()),
                captions_sha256=hashlib.sha256(captions.read_bytes()).hexdigest())
    if penalties:
        data['penalty'] = torch.cat(penalties)
    torch.save(data, output)


class LocalScore(nn.Module):
    def __init__(self, direction, q_dim, v_dim):
        super().__init__()
        self.query = nn.Linear(q_dim, 128)
        self.mean = nn.Linear(v_dim, 128)
        self.delta = nn.Linear(v_dim, 128, bias=False) if direction else None

    def forward(self, q, means, differences):
        visual = self.mean(F.normalize(means, dim=-1))
        if self.delta is not None:
            visual = visual + self.delta(F.normalize(differences, dim=-1))
        query = F.normalize(self.query(q), dim=-1)
        similarity = (F.normalize(visual, dim=-1) * query[:, None, None, :]).sum(-1)
        return (similarity * 10).logsumexp(-1) - math.log(6)


def read(path):
    data = torch.load(path, weights_only=True, map_location='cpu')
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}


def score_batch(model, data, rows, train):
    candidates = data['candidates'][rows]
    base = data['base'][rows]
    labels = data['truth'][rows, None]
    if train:
        missing = (candidates != labels).all(1)
        candidates = candidates.clone()
        base = base.clone()
        candidates[missing, -1] = labels[missing, 0]
        base[missing, -1] = data['true_score'][rows][missing]
    residual = model(data['query'][rows], data['means'][candidates], data['delta'][candidates])
    # A residual of zero recovers first-stage order.
    return base / 0.04 + residual, candidates, labels


@torch.no_grad()
def evaluate(model, data):
    scores = []
    model.eval()
    for rows in torch.arange(len(data['rank']), device='cuda').split(128):
        logits, _, _ = score_batch(model, data, rows, False)
        scores.append(logits)
    return torch.cat(scores)


def ranks(scores, data):
    order = scores.argsort(dim=1, descending=True, stable=True)
    sorted_ids = data['candidates'].gather(1, order)
    hit = sorted_ids == data['truth'][:, None]
    result = torch.where(hit.any(1), hit.int().argmax(1) + 1, data['rank'])
    for k in (20, 100):
        assert torch.equal(result <= k, data['rank'] <= k)
    return result


def metrics(rank, baseline):
    result = {f'R@{k}': round(100 * (rank <= k).sum().item() / len(rank), 4)
              for k in (1, 5, 10, 20, 100)}
    result.update(gain=((rank == 1) & (baseline > 1)).sum().item(),
                  lost=((rank > 1) & (baseline == 1)).sum().item())
    return result


def train(train_data, dev, output):
    assert train_data['checkpoint'] == dev['checkpoint']
    assert set(train_data['vids']).isdisjoint(dev['vids'])
    results = {'baseline': metrics(dev['rank'], dev['rank'])}
    saved = {}
    for direction in (False, True):
        name = 'direction' if direction else 'mean'
        all_logits = []
        for seed in (0, 1, 2):
            set_seed(seed)
            model = LocalScore(direction, train_data['query'].shape[-1], train_data['means'].shape[-1]).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            for _ in range(8):
                model.train()
                ordering = torch.randperm(len(train_data['rank']), device='cuda')
                for rows in ordering.split(96):
                    logits, candidates, labels = score_batch(model, train_data, rows, True)
                    loss = F.cross_entropy(logits, (candidates == labels).long().argmax(1))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    opt.step()
            torch.save(model.state_dict(), output / f'{name}-{seed}.pt')
            logits = evaluate(model, dev)
            all_logits.append(logits)
            results[f'{name}-{seed}'] = metrics(ranks(logits, dev), dev['rank'])
        rank = ranks(torch.stack(all_logits).mean(0), dev)
        saved[name] = rank.tolist()
        results[name] = metrics(rank, dev['rank'])
    with (output/'dev-ranks.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['query', 'baseline', *saved])
        writer.writerows(zip(dev['ids'], dev['rank'].tolist(), *saved.values()))
    (output/'dev.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def test(data, heads, name, strength):
    logits = []
    for seed in (0, 1, 2):
        model = LocalScore(name == 'direction', data['query'].shape[-1], data['means'].shape[-1]).cuda()
        model.load_state_dict(torch.load(heads / f'{name}-{seed}.pt', weights_only=True))
        logits.append(evaluate(model, data))
    original = data['base'] / 0.04
    rank = ranks(original + strength * (torch.stack(logits).mean(0) - original), data)
    with (heads / f'test-{name}-ranks.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['query', 'baseline', name])
        writer.writerows(zip(data['ids'], data['rank'].tolist(), rank.tolist()))
    print(json.dumps({'baseline': metrics(data['rank'], data['rank']),
                      name: metrics(rank, data['rank'])}, indent=2))


def calibrate(data, strength):
    scores = data['base'] - strength * data['penalty'][data['candidates']]
    rank = ranks(scores, data)
    return metrics(rank, data['rank']), rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_subparsers(dest='command', required=True)
    p = group.add_parser('prepare')
    for key in ('checkpoint', 'captions', 'output'):
        p.add_argument(key, type=Path)
    p.add_argument('--bank', type=Path)
    p = group.add_parser('train')
    for key in ('train_cache', 'dev_cache', 'output'):
        p.add_argument(key, type=Path)
    p = group.add_parser('test')
    p.add_argument('test_cache', type=Path)
    p.add_argument('heads', type=Path)
    p.add_argument('variant', choices=('mean', 'direction'))
    p.add_argument('--strength', type=float, default=1.)
    p = group.add_parser('calibrate')
    p.add_argument('cache', type=Path)
    p.add_argument('strength', type=float)
    p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.checkpoint, args.captions, args.output, args.bank)
    elif args.command == 'calibrate':
        data = read(args.cache)
        result, rank = calibrate(data, args.strength)
        with args.output.open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['query', 'baseline', 'calibrated'])
            writer.writerows(zip(data['ids'], data['rank'].tolist(), rank.tolist()))
        print(json.dumps({'strength': args.strength, 'baseline': metrics(data['rank'], data['rank']),
                          'calibrated': result}))
    elif args.command == 'test':
        test(read(args.test_cache), args.heads, args.variant, args.strength)
    else:
        args.output.mkdir(exist_ok=False)
        train(read(args.train_cache), read(args.dev_cache), args.output)

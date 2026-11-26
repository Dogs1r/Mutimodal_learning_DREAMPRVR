"""Fit local visual alignment on training videos and evaluate native CLIP query fusion."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from caption_memory import (baseline_scores, encode_queries, measure, norm, predict, ridge,
                            score, standardize, visual_features)
from Models.builder import get_models
from Utils.utils import load_ckpt


def text_vectors(path, ids):
    with h5py.File(path) as f:
        vectors = []
        for key in ids:
            value = f[key][...]
            if value.ndim == 2 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 1:
                raise ValueError(f'Expected one pooled CLIP text vector for {key}')
            vectors.append(value)
    return norm(torch.from_numpy(np.stack(vectors)).float().cuda())


def fit_maps(train, visual, text):
    captions = text_vectors(text, train['ids'])
    query = norm(train['query'].cuda())
    # Pool each training video once. Bound per-caption window expansion on GPU.
    support = torch.empty_like(captions)
    by_video = [[] for _ in train['vids']]
    for idx, vid in enumerate(train['truth'].tolist()):
        by_video[vid].append(idx)
    for start in range(0, len(by_video), 64):
        windows = visual_features(visual, train['vids'][start:start + 64])
        for local, indices in enumerate(by_video[start:start + 64]):
            attention = (captions[indices] @ windows[local].T * 60).softmax(-1)
            support[indices] = norm(attention @ windows[local])
    return ridge(query, captions), ridge(support, query), ridge(captions, support), ridge(support, captions)


def shortlist_rank(first, second, truth, k=5):
    # Only the order inside the shortlist can change.
    rank, _ = measure(first, truth)
    order = first.argsort(descending=True, dim=1)[:, :k]
    refined = order.gather(1, second.gather(1, order).argsort(descending=True, dim=1))
    matches = refined == truth[:, None]
    return torch.where(matches.any(1), matches.int().argmax(1) + 1, rank)


def rerank_top5(retrieval, refinement):
    """Return a full gallery order with the first five videos rescored."""
    order = retrieval.argsort(dim=1, descending=True, stable=True)
    top = order[:, :5]
    within = refinement.gather(1, top).argsort(dim=1, descending=True, stable=True)
    return torch.cat((top.gather(1, within), order[:, 5:]), dim=1)


def metrics(rank):
    result = {f'R@{k}': (rank <= k).sum().item() * 100 / len(rank) for k in (1, 5, 10, 100)}
    return {**result, 'SumR': sum(result.values())}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('dev', 'test'))
    for name in ('checkpoint', 'data-root', 'train', 'eval', 'visual', 'train-text',
                 'query-text', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--selection', type=Path, help='Locked development selection for testing')
    parser.add_argument('--two-stage', action='store_true',
                        help='Use the fixed raw-CLIP shortlist and trained local refinement')
    args = parser.parse_args()
    if args.phase == 'test' and not args.selection and not args.two_stage:
        parser.error('Testing requires --selection')
    torch.set_num_threads(2)
    cfg, weights, _, _, _ = load_ckpt(args.checkpoint)
    cfg['data_root'] = str(args.data_root)
    root = args.data_root / cfg['collection']
    model = get_models(cfg).cuda().eval()
    model.load_state_dict(weights)
    train, data = (encode_queries(root, path, args.checkpoint, cfg, model)
                   for path in (args.train, args.eval))
    if set(train['vids']) & set(data['vids']):
        raise ValueError('Training and evaluation video IDs overlap')
    query_map, video_map, native_map, native_reverse = fit_maps(train, args.visual, args.train_text)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'query_map': [t.cpu() for t in query_map], 'video_map': [t.cpu() for t in video_map],
                'native_map': [t.cpu() for t in native_map], 'native_reverse': [t.cpu() for t in native_reverse]},
               args.output / 'maps.pt')
    native_q = text_vectors(args.query_text, data['ids'])
    q = norm(data['query'].cuda())
    base = standardize(baseline_scores(cfg, model, data, data['query'].cuda()))
    projected_q = predict(q, query_map)
    mapped_native = predict(native_q, native_map)
    chunks = []
    for start in range(0, len(data['vids']), 256):
        windows = visual_features(args.visual, data['vids'][start:start + 256])
        chunks.append((score(projected_q, windows), score(q, predict(windows, video_map)),
                       score(native_q, windows), score(mapped_native, windows),
                       score(native_q, predict(windows, native_reverse))))
    direct, reverse, native, native_forward, native_back = (standardize(torch.cat(parts, dim=1)) for parts in zip(*chunks))
    aligned = base + .125 * (direct + reverse) / 2
    single = base + .125 * direct
    truth = data['truth'].cuda()
    ranks = {name: measure(value, truth)[0] for name, value in
             [('baseline', base), ('single', single), ('aligned', aligned), ('native_only', native)]}
    ranks['previous_top5'] = shortlist_rank(aligned, single, truth)
    branches = {'raw': native, 'trained': (native_forward + native_back) / 2}
    if args.two_stage:
        # Both scores use the previous development weight, fixed for this experiment.
        weight = .125
        retrieval = aligned + weight * native
        refinement = single + weight * branches['trained']
        order = rerank_top5(retrieval, refinement)
        ranks['retrieval'] = measure(retrieval, truth)[0]
        ranks['last_selected'] = shortlist_rank(aligned + weight * branches['trained'], refinement, truth)
        ranks['selected'] = (order == truth[:, None]).int().argmax(1) + 1
        with (args.output / 'top5.tsv').open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['query', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'])
            for query_id, indices in zip(data['ids'], order[:, :5].tolist()):
                writer.writerow([query_id, *(data['vids'][i] for i in indices)])
        selection = dict(collection=cfg['collection'], weight=weight, top5=True,
                         retrieval='raw', refinement='trained')
        trial_summary = []
    else:
        if args.phase == 'dev':
            grid = [(branch, a, top5) for branch in branches for a in (.125, .25, .5, 1., 2.) for top5 in (False, True)]
        else:
            choice = json.loads(args.selection.read_text())
            if choice['collection'] != cfg['collection']:
                raise ValueError('Selection and checkpoint datasets differ')
            grid = [(choice['branch'], choice['weight'], choice['top5'])]
        trials = []
        for branch, weight, top5 in grid:
            extra = branches[branch]
            combined = aligned + weight * extra
            rank = (shortlist_rank(combined, single + weight * extra, truth) if top5
                    else measure(combined, truth)[0])
            trials.append((metrics(rank), weight, top5, rank, branch))
        best, weight, top5, ranks['selected'], branch = max(
            trials, key=lambda t: (t[0]['R@1'], t[0]['SumR'], not t[2], -t[1]))
        ranks['base_plus_native'] = measure(base + weight * native, truth)[0]
        ranks['aligned_plus_native'] = measure(aligned + weight * native, truth)[0]
        ranks['trained_without_top5'] = measure(aligned + weight * branches['trained'], truth)[0]
        selection = dict(collection=cfg['collection'], weight=weight, top5=top5, branch=branch)
        trial_summary = [dict(metrics=m, weight=w, top5=k, branch=b) for m, w, k, _, b in trials]
    results = {name: metrics(rank) for name, rank in ranks.items()}
    metadata = dict(train_videos=len(train['vids']), train_queries=len(train['ids']),
                    videos=len(data['vids']), queries=len(data['ids']),
                    inputs={key: str(value) for key, value in vars(args).items()})
    for name, obj in [('metrics', results), ('selection', selection), ('metadata', metadata),
                      ('trials', trial_summary)]:
        (args.output / f'{name}.json').write_text(json.dumps(obj, indent=2))
    with (args.output / 'ranks.tsv').open('w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['query', *ranks])
        writer.writerows(zip(data['ids'], *(r.tolist() for r in ranks.values())))
    print(json.dumps(dict(selection=selection, metrics=results), indent=2))


if __name__ == '__main__':
    main()

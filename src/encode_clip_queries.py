"""Encode the supplied query sentences with the OpenAI ViT-B/32 text tower."""
import argparse
from pathlib import Path

import h5py
import torch
from tokenizers import Tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('captions', type=Path, nargs='+')
    args = parser.parse_args()
    sentences = {}
    for path in args.captions:
        for line in path.read_text().splitlines():
            key, sentence = line.split(' ', 1)
            if sentences.setdefault(key, sentence) != sentence:
                raise ValueError(f'Conflicting text for {key}')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer.enable_truncation(max_length=77)
    tokenizer.enable_padding(length=77, pad_id=0, pad_token='!')
    model = torch.jit.load(str(args.weights), map_location='cuda').eval()
    ids = list(sentences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.h5')
    with h5py.File(partial, 'w') as features:
        for start in range(0, len(ids), 128):
            batch = ids[start:start + 128]
            tokens = tokenizer.encode_batch([sentences[i] for i in batch])
            token_ids = torch.tensor([tokens_i.ids for tokens_i in tokens], device='cuda')
            vectors = model.encode_text(token_ids).float().cpu().numpy()
            for key, vector in zip(batch, vectors):
                features.create_dataset(key, data=vector[None])
    partial.replace(args.output)
    print(f'{len(ids)} queries encoded to {args.output}')


if __name__ == '__main__':
    main()

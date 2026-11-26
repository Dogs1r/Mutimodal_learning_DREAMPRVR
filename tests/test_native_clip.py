import unittest

import torch

from native_clip import rerank_top5


class NativeClipRerankTests(unittest.TestCase):
    def test_refinement_only_reorders_top_five(self):
        retrieval = torch.tensor([[9., 8., 7., 6., 5., 4., 3., 2.]])
        refinement = torch.tensor([[0., 1., 6., 2., 3., 100., 100., 100.]])

        order = rerank_top5(retrieval, refinement)
        ranks = order.argsort(dim=1) + 1

        self.assertEqual(order[0, :5].tolist(), [2, 4, 3, 1, 0])
        self.assertEqual(order[0, 5:].tolist(), [5, 6, 7])
        self.assertEqual((ranks <= 5).sum().item(), 5)
        self.assertEqual((ranks <= 8).sum().item(), 8)

    def test_ties_preserve_retrieval_order(self):
        retrieval = torch.tensor([[3., 2., 1., 0., -1., -2.]])
        refinement = torch.zeros_like(retrieval)

        self.assertEqual(rerank_top5(retrieval, refinement)[0, :5].tolist(),
                         [0, 1, 2, 3, 4])

import unittest

import torch

from Losses.loss import query_similarity_preservation_loss


class QuerySimilarityTests(unittest.TestCase):
    def test_antipodal_pair(self):
        loss_fn = query_similarity_preservation_loss({'deterministic_tau': 0.01})
        queries = torch.tensor([[1., 0.], [-1., 0.]], requires_grad=True)
        loss = loss_fn(queries, [0, 0])
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(queries.grad).all())

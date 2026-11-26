import unittest

import torch

from Models.DreamPRVR.event_slots import EventWindows, event_supervision


class EventSlotTests(unittest.TestCase):
    def test_targets_follow_sentences_and_ignore_empty_slots(self):
        slots = torch.eye(6).expand(2, -1, -1)
        queries = torch.eye(6)[[0, 3, 1]]
        labels = [0, 0, 1]
        _, targets, weights = event_supervision(queries, slots, labels, 0.1)
        _, permuted, masses = event_supervision(queries[[2, 0, 1]], slots, [1, 0, 0], 0.1)
        torch.testing.assert_close(weights.sum(1), torch.ones(2))
        torch.testing.assert_close(targets, permuted)
        torch.testing.assert_close(weights, masses)
        self.assertGreaterEqual(int((weights[1] == 0).sum()), 4)
        errors = torch.zeros_like(weights).masked_fill(weights == 0, 100)
        self.assertEqual((weights * errors).sum().item(), 0)
        self.assertFalse(targets.requires_grad)

    def test_local_retrieval_trains_window_centers_and_widths(self):
        torch.manual_seed(3)
        windows = EventWindows(16, 6, learned=True)
        slots, centers, widths = windows(torch.randn(2, 32, 16), torch.randn(2, 6, 16))
        self.assertEqual(slots.shape, (2, 6, 16))
        self.assertTrue((widths >= 1 / 32).all() and (centers < 1).all())
        labels = torch.tensor([0, 0, 1])
        logits, _, _ = event_supervision(torch.randn(3, 16), slots, labels, 0.1)
        torch.nn.functional.cross_entropy(logits / 0.07, labels).backward()
        gradient = windows.offsets.weight.grad
        self.assertTrue(gradient.isfinite().all() and (gradient.norm(dim=1) > 0).all())

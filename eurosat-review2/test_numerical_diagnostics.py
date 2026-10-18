import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from data import EuroSATDataset
from numerical_diagnostics import install


class NumericalDiagnosticsTests(unittest.TestCase):
    def test_healthy_training_has_identical_updates_metrics_and_rng(self):
        torch.manual_seed(73)
        inputs, labels = torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1])
        initial = nn.Linear(4, 2)
        baseline, instrumented = copy.deepcopy(initial), copy.deepcopy(initial)
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        optimizer = torch.optim.AdamW(baseline.parameters(), lr=.001)
        rng = torch.get_rng_state()
        expected = train.run_epoch(baseline, loader, torch.device('cpu'), ['a', 'b'],
                                   optimizer, nn.CrossEntropyLoss())
        expected_rng = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as temp:
            undo = install(train, temp, ['test'])
            try:
                torch.set_rng_state(rng)
                optimizer = torch.optim.AdamW(instrumented.parameters(), lr=.001)
                actual = train.run_epoch(instrumented, loader, torch.device('cpu'), ['a', 'b'],
                                         optimizer, nn.CrossEntropyLoss())
                self.assertEqual(expected, actual)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for name, value in baseline.state_dict().items():
                    self.assertTrue(torch.equal(value, instrumented.state_dict()[name]), name)
            finally:
                undo()

    def test_validation_is_eval_unweighted_and_captures_before_raising(self):
        class FailIfUsed(nn.Module):
            def forward(self, logits, labels):
                raise AssertionError('Validation must not call training criterion')
        class NonfiniteOutput(nn.Module):
            def forward(self, inputs):
                return inputs * float('nan')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index in range(2):
                Image.new('RGB', (2, 2), (20 + index, 40, 60)).save(root / f'{index}.png')
            from torchvision.transforms import ToTensor
            records = [dict(path=f'{i}.png', label=i) for i in range(2)]
            dataset = EuroSATDataset(root, records, ToTensor())
            loader = DataLoader(dataset, batch_size=2)
            model = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(12), nn.Linear(12, 2), NonfiniteOutput())
            model.train()
            undo = install(train, root / 'output', ['test'])
            try:
                with self.assertRaisesRegex(FloatingPointError, 'Non-finite loss'):
                    train.run_epoch(model, loader, torch.device('cpu'), ['a', 'b'],
                                    criterion=FailIfUsed())
                self.assertFalse(model.training)
                captures = list((root / 'output' / '_diagnostics').glob('nonfinite-*'))
                self.assertEqual(len(captures), 1)
                meta = json.loads((captures[0] / 'capture.json').read_text())
                self.assertEqual([r['path'] for r in meta['image_ids']], ['0.png', '1.png'])
                self.assertEqual(meta['targets'], [0, 1])
                self.assertFalse(meta['model_training'])
                self.assertEqual(meta['inputs']['nonfinite'], 0)
                self.assertGreater(meta['logits']['nonfinite'], 0)
                self.assertEqual(meta['nonfinite_parameters'], [])
                self.assertIn('1', meta['batchnorm'])
                self.assertFalse(meta['batchnorm']['1']['training'])
                self.assertEqual(meta['loss_configuration']['criterion_type'], 'unweighted_cross_entropy')
                payload = torch.load(captures[0] / 'batch_and_state.pt', weights_only=True)
                self.assertEqual(payload['inputs'].shape, (2, 3, 2, 2))
                self.assertEqual(payload['unreduced_loss'].shape, (2,))
                self.assertIn('model_state', payload)
                self.assertIn('torch_rng_state', payload)
            finally:
                undo()


if __name__ == '__main__':
    unittest.main()

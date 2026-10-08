"""Small protocol tests; no PyTorch or downloaded dataset required."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiment_utils import save_json, select_size, validate_manifest
from metrics import classification_metrics
from summarize_results import summarize


class ProtocolTests(unittest.TestCase):
    def test_resolution_selection(self):
        self.assertEqual(select_size({224:[.98]*3,96:[.979]*3,64:[.974]*3}),96)
        self.assertEqual(select_size({224:[.98]*3,96:[.97]*3,64:[.96]*3}),224)
        self.assertEqual(select_size({224:[.98]*3,96:[.981]*3,64:[.98]*3}),64)

    def test_manifest_overlap(self):
        manifest={'classes':[str(i) for i in range(10)],'splits':{
            split:[{'path':f'{split}/{i}.jpg','label':i} for i in range(10)] for split in ['train','val','test']}}
        validate_manifest(manifest)
        broken=copy.deepcopy(manifest)
        broken['splits']['val'][0]['path']='train/0.jpg'
        with self.assertRaises(ValueError): validate_manifest(broken)

    def test_metrics_and_sample_std(self):
        result=classification_metrics([0,0,1,1],[0,1,1,1],['a','b'])
        self.assertAlmostEqual(result['balanced_accuracy'],.75)
        self.assertAlmostEqual(result['per_class_recall']['a'],.5)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for seed,score in [(42,.8),(123,.9),(2024,1.0)]:
                folder=root/str(seed);folder.mkdir()
                save_json(folder/'config.json',dict(seed=seed,image_size=64,norm='batch',initialization='imagenet',split_hash='same'))
                save_json(folder/'status.json',dict(status='completed',best_epoch=2,epochs_completed=3,stopping_reason='max_epochs',training_loop_seconds=10))
                save_json(folder/'best_validation_metrics.json',dict(accuracy=score,macro_f1=score,balanced_accuracy=score,per_class_recall={'a':score}))
            summarize(root)
            import csv
            with (root/'summary.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]['val_macro_f1_mean']),.9)
            self.assertAlmostEqual(float(rows[0]['val_macro_f1_std']),.1)
            self.assertNotIn('test_macro_f1_mean',rows[0])
            self.assertIn('pending',(root/'phase1_results.tex').read_text())


if __name__=='__main__': unittest.main()

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from losses import effective_weights, ClassBalancedLoss, FocalLoss
from partition import generate, validate_partition, freeze, label_dirichlet, SEEDS, ALPHAS, RATIOS
from experiment_utils import read_json
from review_workflow import gn_jobs,tuning_jobs,final_jobs,Workflow
from results_review import mean_sd

BASE=Path(__file__).resolve().parent

class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.manifest=read_json(BASE/'splits.json')

    def test_all_partitions_are_reproducible_and_disjoint(self):
        for seed in SEEDS:
            for alpha in ALPHAS:
                a=generate(self.manifest,'dirichlet',seed,alpha=alpha)
                b=generate(self.manifest,'dirichlet',seed,alpha=alpha)
                self.assertEqual(a,b);validate_partition(self.manifest,a)
                self.assertEqual(sum(map(len,a['clients'].values())),18900)
                self.assertGreaterEqual(min(map(len,a['clients'].values())),10)

    def test_longtail_exact_counts_fixed_order(self):
        orders=[]
        for seed in SEEDS:
            for ratio in RATIOS:
                p=generate(self.manifest,'longtail',seed,ratio=ratio)
                orders.append(p['class_order'])
                expected=[int(2100*ratio**(-k/9)) for k in range(10)]
                self.assertEqual(p['rank_counts'],expected)
                self.assertEqual(p['realized_ratio'],ratio)
                validate_partition(self.manifest,p)
        self.assertTrue(all(o==orders[0] for o in orders))

    def test_partition_blocks_eval_leakage_and_tampering(self):
        p=generate(self.manifest,'dirichlet',42,alpha=.1)
        p['clients']['0'][0]=self.manifest['splits']['test'][0]['path']
        with self.assertRaises(ValueError):validate_partition(self.manifest,p)
        with self.assertRaises(ValueError):label_dirichlet([0]*9,.1,10,42)
        with self.assertRaises(RuntimeError):label_dirichlet([0]*100,.1,10,42,max_attempts=0)

    def test_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'p.json';freeze(path,dict(b=2,a=1));before=path.read_bytes()
            freeze(path,dict(a=1,b=2));self.assertEqual(before,path.read_bytes())
            with self.assertRaises(ValueError):freeze(path,dict(a=2))

    def test_losses_equal_ce_in_limiting_cases(self):
        torch.manual_seed(1);logits=torch.randn(5,3,requires_grad=True);labels=torch.tensor([0,1,2,1,0])
        ce=F.cross_entropy(logits,labels)
        self.assertTrue(torch.allclose(ce,FocalLoss(0)(logits,labels)))
        self.assertTrue(torch.allclose(ce,ClassBalancedLoss([10,10,10],.9999)(logits,labels)))
        self.assertTrue(torch.allclose(effective_weights([1,10,100],0),torch.ones(3)))

    def test_absent_classes_and_extreme_logits(self):
        w=effective_weights([0,1,100],.9999);self.assertEqual(w[0].item(),0);self.assertGreater(w[1],w[2])
        for loss in [ClassBalancedLoss([0,1,100],.9999),FocalLoss(2)]:
            logits=torch.tensor([[10000.,-10000.,0.],[-10000.,0.,10000.]],requires_grad=True)
            value=loss(logits,torch.tensor([1,2]));value.backward()
            self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(logits.grad).all())
        with self.assertRaises(ValueError):effective_weights([0,0],.999)

    def test_run_plan_reuses_tuning(self):
        choices={k:dict(beta=.999,gamma=2.) for k in ['dirichlet','longtail']}
        choices['gn_comparison']=dict(loss='class-balanced',beta=.999,gamma=0.)
        final=final_jobs(.001,choices)
        self.assertEqual(len(final),680);self.assertEqual(len(gn_jobs()),9);self.assertEqual(len(tuning_jobs()),774)
        unique={json.dumps(j,sort_keys=True) for j in final+gn_jobs()+tuning_jobs()}
        self.assertEqual(len(unique),1073)

    def test_client_variation_is_not_seed_uncertainty(self):
        rows=[]
        for seed,offset in zip(SEEDS,[.1,.2,.3,.4,.5]):
            for cid in range(10):rows.append(dict(seed=seed,score=offset+cid*.001))
        frame=pd.DataFrame(rows);m,sd=mean_sd(frame,'score')
        self.assertAlmostEqual(m,.3045);self.assertAlmostEqual(sd,np.std([.1,.2,.3,.4,.5],ddof=1))
        with self.assertRaises(ValueError):mean_sd(frame[frame.seed!=99],'score')

    def test_test_access_blocks_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(output=temp,split=str(BASE/'splits.json'),epochs=100)
            workflow=Workflow(args)
            d=Path(temp)/'runs'/'fake';d.mkdir(parents=True);(d/'test_metrics.json').write_text('{}')
            with self.assertRaises(ValueError):workflow.select_gn()
            with self.assertRaises(ValueError):workflow.select_losses()

if __name__=='__main__':unittest.main(verbosity=2)

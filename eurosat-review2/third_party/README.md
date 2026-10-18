# NIID-Bench adaptation

`partition.py:label_dirichlet` adapts the `noniid-labeldir` branch of:
https://github.com/Xtra-Computing/NIID-Bench/blob/61b901ab645e62391772e9898384c2c3485b7a6c/partition.py

Upstream commit: 61b901ab645e62391772e9898384c2c3485b7a6c.
The MIT notice is retained in NIID_Bench_LICENSE.

Preserved: per-class shuffling, symmetric Dirichlet sampling, client-size
balancing mask, cumulative integer cuts, minimum 10 samples, final shuffle.
Changes: local RandomState, bounded retry loop, underflow safeguard, explicit
argument validation, relative image paths, training-only manifests and hashes.
Retrying uses a continuing seeded RNG stream rather than resetting the seed.

Cite Qinbin Li, Yiqun Diao, Quan Chen and Bingsheng He, "Federated Learning on
Non-IID Data Silos: An Experimental Study," ICDE 2022; arXiv:2102.02079.
This project trains local-only client models. It does not implement FedAvg.

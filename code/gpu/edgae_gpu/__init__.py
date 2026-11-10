"""EDGaE native GPU experiments (pure PyTorch, no DGL dependency).

Targets the paper's missing native evidence:
  E1  full-batch GCN reference accuracy on ogbn-arxiv (official split);
  E2  mini-batch GraphSAGE, server-centric vs. EDGaE feature paths;
  E3  preprocessing-inclusive time-to-target;
  E4  sampler fan-out sensitivity at the native level.
"""

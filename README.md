This repository contains the source code of the poisson inverse problem experiment in the paper "Fast minimization of expected logarithmic loss via stochastic dual averaging" accepted by AISTATS 2024.

# How to run
- Tested on [Julia](https://julialang.org) Version 1.9.2
- Set the dimension and the number of samples in line 23 to 25 in `main.jl`
## Install Packages
```
$ cd pip/
$ julia ./install.jl
```
## Run
```
$ julia ./main.jl
```

# Implemented Algorithms
## Batch Algorithms
1. Expectation maximization (EM): L. A. Shepp and Y. Vardi, Maximum likelihood reconstruction for emission tomography, *IEEE Trans. Med. Imaging*, 1982 ([link](https://ieeexplore.ieee.org/abstract/document/4307558?casa_token=buwWKAGzPQoAAAAA:O6IqktAyIfScoGkC0Q0jYXQZsJUNCnXg1jkZHQ6WwoduwPjI2EwIU1ef2WiuesmSYc4qhJYsVg)) and Thomas M. Cover, An algorithm for maximizing expected log investment return, *IEEE Trans. Inf. Theory*, 1984 ([link](https://ieeexplore.ieee.org/abstract/document/1056869?casa_token=y0cA70bABs0AAAAA:zpsP7RfwrwwlhCno5liSw3OU1Fha6yFDdJk9UDvLCSAMnC0teguSlZx31ILbUc1SoVk0bep2yw))
2. Primal-dual hybrid gradient method (PDHG): Antonin Chambolle and Thomas Pock, A first-order primal-dual algorithm for convex problems with applications to imaging, *J. Math. Imaging Vis.*, 2011 ([link](https://link.springer.com/article/10.1007/s10851-010-0251-1))
3. NoLips: Heinz H. Bauschke, Jérôme Bolte, Marc Teboulle, A descent lemma beyond Lipschitz gradient continuity: first-order methods revisited and applications, *Math. Oper. Res.*, 2017 ([link](https://pubsonline.informs.org/doi/abs/10.1287/moor.2016.0817))
4. Entropic mirror descent with Armijo line search (EMD): Yen-Huan Li and Volkan Cevher, Convergence of the exponentiated gradient method with Armijo line search, *J. Optim. Theory Appl.*, 2019 ([link](https://link.springer.com/article/10.1007/s10957-018-1428-9))
5. Frank-Wolfe (FW): Renbo Zhao and Robert M. Freund, Analysis of the Frank–Wolfe method for convex composite optimization involving a logarithmically-homogeneous barrier, *Math. Program.*, 2023 ([link](https://link.springer.com/article/10.1007/s10107-022-01820-9)) 
## Stochastic Algorithms
1. Stochastic primal-dual hybrid gradient (SPDHG): Antonin Chambolle, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schönlieb, Stochastic primal-dual hybrid gradient algorithm with arbitrary sampling and imaging applications, *SIAM J. Optim.*, 2018 ([link](https://epubs.siam.org/doi/abs/10.1137/17M1134834))
2. Stochastic Soft-Bayes (SSB): Yen-Huan Li, Online positron emission tomography By online portfolio selection, *Int. Conf. Acoustics, Speech, and Signal Processing (ICASSP)*, 2020 ([link](https://ieeexplore.ieee.org/abstract/document/9053230))
3. Stochastic LB-OMD (SLBOMD): Chung-En Tsai, Hao-Chung Cheng, and Yen-Huan Li, Faster stochastic first-order method for maximum-likelihood quantum state tomography, *Int. Conf. Quantum Information Processing (QIP)*, 2023 ([link](https://arxiv.org/abs/2211.12880))
4. 1-sample LB-SDA: **this work**
5. $d$-sample LB-SDA: **this work**

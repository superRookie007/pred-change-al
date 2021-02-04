# pred-change-al
Implementation of computing prediction changes during training neural networks for active learning. This is inspired by recent research on the convergence property of neural networks. See below for some references.

Toneva, Mariya, et al. "An empirical study of example forgetting during deep neural network learning." arXiv preprint arXiv:1812.05159 (2018).

Nacson, Mor Shpigel, et al. "Convergence of gradient descent on separable data." The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.

Soudry, Daniel, et al. "The implicit bias of gradient descent on separable data." The Journal of Machine Learning Research 19.1 (2018): 2822-2878.

Xu, Tengyu, et al. "When will gradient methods converge to max‐margin classifier under ReLU models?." Stat (2018).


The cifar10_resnet_main.py is an example of how to perform active learning by accumulating prediction changes during training on CIFAR-10 using resnet18. The bash file example.sh shows an example command along with explanation of the arguments. 

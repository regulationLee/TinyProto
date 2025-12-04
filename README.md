# TinyProto
Code for the paper **"Communication-Efficient Heterogeneous Federated Learning with Sparse Prototypes in Resource-Constrained Environments"** a.k.a TinyProto, accepted to AAAI 2026 (Main)

Communication efficiency in federated learning (FL) remains a critical challenge in resource-constrained environments. While prototype-based FL reduces communication overhead by sharing class prototypes—mean activations 
in the penultimate layer—instead of model parameters, its efficiency degrades with larger feature dimensions and class counts. We propose TinyProto, which addresses these limitations through Class-wise Prototype Sparsification (CPS) 
and Adaptive Prototype Scaling (APS). CPS enables structured sparsity by allocating specific dimensions to class prototypes and transmitting only non-zero elements, thereby achieving higher communication efficiency, 
while APS scales prototypes based on class distributions to improve performance. Our experiments demonstrate that TinyProto reduces communication costs by up to 10×compared to existing methods while improving performance. 
Beyond communication efficiency, TinyProto offers crucial advantages: it achieves compression without client-side computational overhead and supports heterogeneous architectures, making it particularly suitable for 
resource-constrained heterogeneous FL scenarios. (Paper: [arXiv](https://arxiv.org/abs/2507.04327) )


## ⚙️ Implementation
### Ubuntu
```sh
pyenv install 3.10
pyenv local 3.10

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
# if cuda version 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


### Dataset setting
```sh
cd dataset
python generate_cifar10.py noniid - dir
cd ..
```

### run TinyFP
```sh
cd system
python -u main.py -did 0 -data Cifar10 -m Ht0 -algo TinyFP -gr 300 -lam 10 -ssc -csf 0.00015 -go test > test.out 2>&1 &
```

## 📌 Framework Foundation
This repository contains the implementation code for our paper, built upon **[PFLlib: Personalized Federated Learning Library and Benchmark](https://github.com/TsingZ0/PFLlib)**.

PFLlib is a comprehensive Python library featuring 39+ federated learning algorithms, 24 datasets, and specialized support for addressing data heterogeneity in personalized federated learning scenarios.

## 🙏 Acknowledgments
We gratefully acknowledge the PFLlib team for providing an excellent foundation for personalized federated learning research.

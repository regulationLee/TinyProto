# TinyProto
[AAAI '26 Accepted] TinyProto: Communication-Efficient Federated Learning with Sparse Prototypes in Resource-Constrained Environments([arXiv](https://www.arxiv.org/abs/2507.04327))


### Environment setting
#### Ubuntu
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

# Block Sparse Neural Networks

## Training Command
The script `train.py` initiates model training. This script takes the following arguments.

1. Dataset Name: The name of the dataset. Must be one of `mnist`, `fashion_mnist`, `uci_har`, or `function`. The actual data must be located at `datasets/<name>`.
2. Hyperparameters: Path to the hyper parameters file. Most paramers will be found in the `params` folder.
3. Use GPU: Whether to use the GPU. If not provided, then the program will use the available CPU.
4. Log Device: Whether to print out the device name.
5. Should Print: Whether to print out logging information during training.

An example command for the dunny `function` dataset is below.
```
python train.py --dataset function --hypers-file params/function/dense/mlp.json --should-print
```


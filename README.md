# SDA-FC
Implementation of paper ["Federated clustering with GAN-based data synthesis"](https://arxiv.org/abs/2210.16524)

# Requirements:
* Python: 3.8
* Pytorch: 1.11.0
* Packages:
  * `pip install fuzzy-c-means`
  * `pip install scikit-learn`
  * `pip install munkres`

# Training for MNIST
* Run `python main.py --data_root datasets/mnist --exp_dir output/mnist --p 0.25` 

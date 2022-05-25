# Don’t fear the unlabelled: Safe semi-supervised learning via simple debiasing

This repository is an implementation of the debiasing method proposed in  our paper based on an [unofficial pytorch code for Fixmatch](https://github.com/LeeDoYup/FixMatch-pytorch).
This implementation can reproduce the resultson CIFAR-10 of our paper but also on CIFAR-100.


As explained in the paper, we modified the supervised loss of Fixmatch to include strong augmentations:


$$
    L(\theta;x,y) = \frac{1}{2}\left(\mathbb{E}_{x_1\sim\textit{weak}(x)}[-\log(p_{\theta}(y|x_1))] + \mathbb{E}_{x_2\sim\textit{strong}(x)}[-\log(p_{\theta}(y|x_2))]\right),
$$
where $x_1$ is a weak augmentation of $x$ and $x_2$ is a strong augmentation. This modification encourages us to choose $\lambda=\frac{1}{2}$ as the original Fixmatch implementation used $\lambda =1$.
The unsupervised loss of Fixmatch remains unchanged:
$$ H(\theta; x) =\mathbb{E}_{x_1\sim\textit{weak}(x)}\left[ \mathbb{1}[\max_yp_{\hat{\theta}}(y|x_1)>\tau]
\mathbb{E}_{x_2\sim\textit{strong}(x)}[-\log(p_{\theta}(\argmax_yp_{\hat{\theta}}(y|x_1)|x_2))]\right].$$

The training objective for the Complete Case is
$$ 
\hat{\mathcal{R}}_{CC}(\theta) = \frac{1}{n_l}\sum_{i=1}^{n_l}L(\theta; x_i,y_i).
$$

The training objective for Fixmatch is
$$ 
\hat{\mathcal{R}}_{DeSSL}(\theta) = \frac{1}{n_l}\sum_{i=1}^{n_l}L(\theta; x_i,y_i)  \color{red}{+ \frac{\lambda}{n_u}\sum_{i=1}^{n_u}H(\theta; x_i)} .
$$

The training objective for DeFixmatch is
$$ 
\hat{\mathcal{R}}_{DeSSL}(\theta) = \frac{1}{n_l}\sum_{i=1}^{n_l}L(\theta; x_i,y_i)  \color{red}{+ \frac{\lambda}{n_u}\sum_{i=1}^{n_u}H(\theta; x_i)}  \color{blue}{- \frac{\lambda}{n_l}\sum_{i=1}^{n_l}H(\theta; x_i)}.
$$

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Training
We recommend using distributed training for high performance.  
**With V100x4 GPUs, CIFAR10 training takes about 16 hours (0.7 days), and CIFAR100 training takes about 62 hours (2.6 days).**

To train the complete case model on CIFAR-10 with $n_l=4000$, run this command:

```train
python train.py --world-size 1 --rank 0 --multiprocessing-distributed --num_labels 4000 --dataset cifar10 --num_classes 10 --overwrite --modified_fixmatch --ulb_loss_ratio 0
```

To train the Fixmatch on CIFAR-10 with $n_l=4000$, run this command:

```train
python train.py --world-size 1 --rank 0 --multiprocessing-distributed --num_labels 4000 --dataset cifar10 --num_classes 10 --overwrite --modified_fixmatch --ulb_loss_ratio 0.5
```

To train the Fixmatch on CIFAR-10 with $n_l=4000$, run this command:

```train
python train.py --world-size 1 --rank 0 --multiprocessing-distributed --num_labels 4000 --dataset cifar10 --num_classes 10 --overwrite --debiased --ulb_loss_ratio 0.5
```

**Trained models are saved in the directory: saved_models/**


## Evaluation

To evaluate my model on CIFAR-10 using a checkpoint, run:

```eval
python eval.py --load_path model.pth --dataset cifar10 --num_classes 10

```

## Pre-trained Models

You can find pretrained models on CIFAR-10 using $n_l=4000$ in the **saved_models/** directory.


## Results

Our models achieves the following performance on CIFAR-10:


| Model name         |  Accuracy  | Cross-entropy | Worst class Accuracy |
| ------------------ |---------------- | -------------- | -------------- |
| Complete Case      |    87.27 $\pm$ 0.25%         |      0.60 $\pm$ 0.01       | 70.08 $\pm$ 0.93% |
| Fixmatch           |     93.87 $\pm$ 0.13%         |      0.27 $\pm$ 0.01       | 82.25 $\pm$ 2.27% |
| DeFixmatch         |     **95.44 $\pm$ 0.10%**         |      **0.20 $\pm$ 0.01**       | **87.16 $\pm$ 0.46** |



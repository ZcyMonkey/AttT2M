# (ICCV 2023) AttT2M

Code of ICCV 2023 paper: "AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism"

[[Paper]](https://arxiv.org/)


<p align="center">
<img src="img/teaser.png" width="600px" alt="teaser">
</p>

## 1. Results

### 1.1 Visual Results

### Compare with SOTA
<p align="center">
<img src="img/compare.gif" width="600px" alt="compare">
</p>

### 1.2 Quantitative Results
 
## 2. Installation

### 2.1. Environment

```bash
conda env create -f environment.yml
conda activate Att-T2M
```
The code was tested on Python 3.8 and PyTorch 1.8.1.


### 2.2. Datasets and others

We use two dataset: HumanML3D and KIT-ML. For both datasets, the details about them can be found [[here]](https://github.com/EricGuo5513/HumanML3D).   
Motion & text feature extractors are also provided by [t2m](https://github.com/EricGuo5513/text-to-motion) to evaluate our generated motions



## 3. Quick Start
```
python vis.py
```

## 4. Train

coming


## 5. Evaluation 

coming

## 6. Acknowledgement

* Part of the code is borrowed from public code like [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse) etc.




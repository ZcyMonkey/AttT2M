# (ICCV 2023) AttT2M

Code of ICCV 2023 paper: "AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism"

[[Paper]](https://arxiv.org/)


<p align="center">
<img src="img/teaser.png" width="600px" alt="teaser">
</p>

## 1. Results

### 1.1 Visual Results

<!-- ![visualization](img/ALLvis_new.png) -->

<p align="center">
<table>
  <tr>
    <th colspan="5">Text: a person quickly waves with their right hand.</th>
  </tr>
  <tr>
    <th><u><a href="https://github.com/Mael-zys/T2M-GPT/"><nobr>T2M-GPT</nobr> </a></u></th>
    <th><u><a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html"><nobr>MotionDiffuse</nobr> </a></u></th>
    <th>Ours</th>
  </tr>
  
  <tr>
    <td><img src="img/002103_pred_t2m_gpt_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_MotionDiffuse_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_16.gif" width="140px" alt="gif"></td>
  </tr>

  <tr>
    <th colspan="5">Text: A person walk in a circle clockwise.</th>
  </tr>
  <tr>
    <th><u><a href="https://github.com/Mael-zys/T2M-GPT/"><nobr>T2M-GPT</nobr> </a></u></th>
    <th><u><a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html"><nobr>MotionDiffuse</nobr> </a></u></th>
    <th>Ours</th>
  </tr>
  
  <tr>
    <td><img src="img/000066_pred_t2m_gpt_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/000066_pred_MotionDiffuse_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/000066_pred_16.gif" width="140px" alt="gif"></td>
  </tr>
</table>
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




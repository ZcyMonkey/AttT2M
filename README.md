# (ICCV 2023) AttT2M

Code of ICCV 2023 paper: "AttT2M: Text-Driven Human Motion Generation with Multi-Perspective Attention Mechanism"

[[Paper]](https://arxiv.org/)


<p align="center">
<img src="img/Teaser.png" width="600px" alt="teaser">
</p>

## Table of Content
* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Quick Start](#3-quick-start)
* [4. Train](#4-train)
* [5. Evaluation](#5-evaluation)
* [6. SMPL Mesh Rendering](#6-smpl-mesh-rendering)
* [7. Acknowledgement](#7-acknowledgement)
* [8. ChangLog](#8-changlog)




## 1. Results

### 1.1 Visual Results

<!-- ![visualization](img/ALLvis_new.png) -->

<p align="center">
<table>
  <tr>
    <th colspan="5">Text: a man steps forward and does a handstand.</th>
  </tr>
  <tr>
    <th>GT</th>
    <th><u><a href="https://ericguo5513.github.io/text-to-motion/"><nobr>T2M</nobr> </a></u></th>
    <th><u><a href="https://guytevet.github.io/mdm-page/"><nobr>MDM</nobr> </a></u></th>
    <th><u><a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html"><nobr>MotionDiffuse</nobr> </a></u></th>
    <th>Ours</th>
  </tr>
  
  <tr>
    <td><img src="img/002103_gt_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_t2m_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_mdm_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_MotionDiffuse_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/002103_pred_16.gif" width="140px" alt="gif"></td>
  </tr>

  <tr>
    <th colspan="5">Text: A man rises from the ground, walks in a circle and sits back down on the ground.</th>
  </tr>
  <tr>
    <th>GT</th>
    <th><u><a href="https://ericguo5513.github.io/text-to-motion/"><nobr>T2M</nobr> </a></u></th>
    <th><u><a href="https://guytevet.github.io/mdm-page/"><nobr>MDM</nobr> </a></u></th>
    <th><u><a href="https://mingyuan-zhang.github.io/projects/MotionDiffuse.html"><nobr>MotionDiffuse</nobr> </a></u></th>
    <th>Ours</th>
  </tr>
  
  <tr>
    <td><img src="img/000066_gt_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/000066_pred_t2m_16.gif" width="140px" alt="gif"></td>
    <td><img src="img/000066_pred_mdm_16.gif" width="140px" alt="gif"></td>
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

TBD


## 5. Evaluation 

TBD

## 6. Acknowledgement

* Part of the code is borrowed from public code like [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse) etc.




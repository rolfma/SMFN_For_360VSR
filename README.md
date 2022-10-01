##  Code Repository for A Single Frame and Multi-Frame Joint Network for 360-degree Panorama Video Super-Resolution (SMFN)

### 1 Prerequisites

#### 1.1 Install PyTorch

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

#### 1.2 Install MMCV

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```

### 2 Data Preparation
```shell
ln -s /your/path data/MigVR
ln -s /your/path data/MigVR56789
ln -s /your/path data/MigVRTest
```

### 3 Train and Test
#### 3.1 Init
```shell
python setup.py develop
```

#### 3.2 Train
```shell
python options/SMFN/train.py
```

#### 3.3 Test
```
python options/SMFN/test.py
```


### Acknowledgements
The organization of this repository is modeled after BasicSR.
```latex
@misc{wang2020basicsr,
  author =       {Xintao Wang and Ke Yu and Kelvin C.K. Chan and
                  Chao Dong and Chen Change Loy},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/xinntao/BasicSR}},
  year =         {2018}
}
```

# Unsupervised Volumetric Animation

This repository contains the source code for the CVPR'2023 paper [Unsupervised Volumetric Animation](https://arxiv.org/abs/2301.11326).
For more qualitiative examples visit our [project page](https://snap-research.github.io/unsupervised-volumetric-animation/).

Here is an example of several images produced by our method. 
On the left is sample visualization: In the first column the driving video is shown. For the remaining columns the top image is animated by using motions extracted from the driving.
On the right is rotation visualization: We show source image as well as rotated rgb, depth, normals and segments. 

Sample animation           |  Rotation visualization
:-------------------------:|:-------------------------:
![](./assets/sample.gif)  |  ![](./assets/rotation.gif)

### Installation

We support ```python3```. To install the dependencies run:
```bash
pip install -r requirements.txt
```

### YAML configs

There are several configuration files one for each `dataset` in the `config` folder named as ```config/dataset_name_stage.yaml```. We adjust the configuration to run on 8 A100 GPU.

### Pre-trained checkpoints
Checkpoints can be found under this [link]().

### Inversion
Inversion, to run inversion on your own image use:
```bash
python inversion.py  --config config/dataset_name.yaml --driving_video path/to/driving --source_image path/to/source --checkpoint tb-logs/vox_second_stage/{time}/checkpoints/last.cpkt
```
The result can be seen with tensorboard.


### Training

To train a model, fist, download the mraa checkpoints and place them into ```./```.
Then run the following commands:

```bash
python train.py --config config/vox_first_stage.yaml
# [Optional] To save time one could first train with 128 resolution:
python train.py --config config/vox_second_stage_128.yaml --checkpoint tb-logs/vox_first_stage/{time}/checkpoints/last.cpkt
python train.py --config config/vox_second_stage.yaml --checkpoint tb-logs/vox_first_stage/{time}/checkpoints/last.cpkt
```

Citation:
```
@article{siarohin2023unsupervised,
    author  = {Siarohin, Aliaksandr and Menapace, Willi and Skorokhodov, Ivan and Olszewski, Kyle and Lee, Hsin-Ying and Ren, Jian and  Chai, Menglei and Tulyakov, Sergey},
    title   = {Unsupervised Volumetric Animation},
    journal = {arXiv preprint arXiv:2301.11326},
    year    = {2023},
}
```

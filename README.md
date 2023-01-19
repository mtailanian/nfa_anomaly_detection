<!-- <div align="center"> -->

<div style="text-align: center;" markdown="1">

# A Contrario multi-scale anomaly detection method for industrial quality inspection

[Link to code](https://www.github.com/mtailanian/nfa_anomaly_detection)

### Book chapter

[![Arxiv](https://img.shields.io/badge/arXiv-2205.11611-blue.svg)](https://arxiv.org/pdf/2205.11611.pdf)

[![Book chapter](https://img.shields.io/badge/Book-Deep%20%20Learning%20Applications%2C%20Volume%204-orange.svg)](https://link.springer.com/chapter/10.1007/978-981-19-6153-3_8)

### Conference paper

[![Paper](https://img.shields.io/badge/ICMLA-2021-yellow.svg)](https://ieeexplore.ieee.org/abstract/document/9680125)
[![Arxiv](https://img.shields.io/badge/arXiv-2110.02407-red.svg)](https://arxiv.org/abs/2110.02407)
 
</div>

## Description   
This is the official code that implements the chapter **A Contrario multi-scale 
anomaly detection method for industrial quality inspection**, of the book **Deep 
Learning Applications, Volume 4**  

The algorithm is completely unsupervised and intends to detect anomalies in textured images.

### Result examples over MVTec-AD Dataset
![text](assets/mvtec.png?raw=true)

For each textured dataset in MVTec AD (carpet, grid, leather, tile, and
wood), we show one image with each type of defect and their corresponding
anomaly maps with the ResNet+RegionNFA method. The ground truth in is
shown green, and the detection with log-NFA=0 in blue, superimposed to the
original image.

### Result examples over MVTec-AD Dataset
![text](assets/results.png?raw=true)

## How to run   
### Setup
First, download repo, create virtual environment and install dependencies   
```bash
# clone project   
git clone https://github.com/mtailanian/nfa_anomaly_detection.git
cd nfa_anomaly_detection

# Create virtualenv and activate it
virtualenv -p python3 .env
source .env/bin/activate

# install dependencies
pip install -r requirements.txt
 ```   

### Run
 Next, run it using `main.py`, and passing the image path. 

A test image is provided in `./images/test_image_01.jpg`

For example:
 ```bash
python main.py images/test_image_01.jpg
```

Other additional optional arguments:

| **Argument short name** | **Argument long name** |                                          **Description**                                           |     **Default value**     |
|:-----------------------:|:----------------------:|:--------------------------------------------------------------------------------------------------:|:-------------------------:|
|       image_path        |       image_path       |                                    Path of the image to process                                    | None (mandatory argument) |
|          -thr           |  --log_nfa_threshold   |                    Threshold over the computed NFA map, for final segmentation.                    |             0             |
|        -dist_thr        |  --distance_threshold  |       Threshold over the squared Mahalanobis distances, for computing the candidate regions.       |            0.5            |
|           -s            |         --size         |                          Input size for ResNet. Must be divisible by 32.                           |            256            |
|          -pca           |       --pca_std        | If float: the percentage of the variance to keep in PCA. If int: the number of components to keep. |            35             |

### Citation

Conference paper:

```
@inproceedings{tailanian2021multi,
  title={A Multi-Scale A Contrario method for Unsupervised Image Anomaly Detection},
  author={Tailani{\'a}n, Mat{\'\i}as and Mus{\'e}, Pablo and Pardo, {\'A}lvaro},
  booktitle={2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA)},
  pages={179--184},
  year={2021},
  organization={IEEE}
}
```   

Book chapter:

```
@incollection{tailanian2023contrario,
  title={A Contrario multi-scale anomaly detection method for industrial quality inspection},
  author={Tailanian, Mat{\'\i}as and Mus{\'e}, Pablo and Pardo, {\'A}lvaro},
  booktitle={Deep Learning Applications, Volume 4},
  pages={193--216},
  year={2023},
  publisher={Springer}
}
```

Copyright and License
---------------------

Copyright (c) 2021-2022 Matias Tailanian <mtailanian@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Introduction
* Python code for our paper (Highly differentiated target detection under extremely low-light conditions based on improved YOLOx model)
* Note: The original implementation of PyTorch version YOLOX, please vist: [MegEngine implementation](https://github.com/MegEngine/YOLOX).

## Updates
* 【2022/01/20】 We suport jit compile op.[SSRN](https://assets.researchsquare.com/files/rs-1212268/v1_covered.pdf?c=1642711187)
* 【2023/10/21】 We optimize the training process under the low-light condition and obtain higher performance! See [notes](Coming soon) for more details.

## TODO: Update with exact module version
numpy,torch>=1.7,opencv_python,loguru,tqdm,torchvision,thop,ninja,tabulate

## Dataset
* Pascal VOC datasets from the VOC challenges are available through the challenge links: [Link](http://host.robots.ox.ac.uk/pascal/VOC/)
* MS COCO 2017 datasets can be found at: [Link](https://cocodataset.org/#home)

## Benchmark
### Table 1: Performance Comparison of Low-Light Image Detection

| Methods          | Proposed | YOLOx | YOLOv4 | RFBnet | Mobilenet-SSD | Faster-RCNN | M2det   |
|------------------|----------|-------|--------|--------|---------------|-------------|---------|
| Dark image       | 70.96    | 67.41 | 54.96  | 64.06  | 53.91         | 65.19       | 65.09   |
| Dong et al.[1]   | 72.15 | 68.47 | 47.75  | 64.09  | 36.52         | 62.71       | 62.20   |
| This paper       | 76.86    | 73.31 | 69.83  | 73.79  | 63.26         | 62.75       | 72.32   |

### Table 2: Analysis of Average Precision (AP) of YOLOx in Various Lighting Conditions

| Image            | AP@0.50:0.95 | AP@0.50 | AP@0.75 | AP@S   | AP@M   | AP@L   |
|------------------|--------------|---------|---------|--------|--------|--------|
| Original image   | 0.504        | 0.690   | 0.547   | 0.325  | 0.561  | 0.669  |
| Dark image       | 0.404        | 0.592   | 0.426   | 0.196  | 0.448  | 0.597  |
| This paper       | 0.456        | 0.643   | 0.489   | 0.239  | 0.508  | 0.650  |

### Table 3: Augmented Reality (AR) of YOLOx in Various Illumination Conditions

| Image            | AR@0.50:0.95 | AR@0.50 | AR@0.75 | AR@S   | AR@M   | AR@L   |
|------------------|--------------|---------|---------|--------|--------|--------|
| Original image   | 0.379        | 0.614   | 0.653   | 0.468  | 0.712  | 0.825  |
| Dark image       | 0.327        | 0.514   | 0.549   | 0.334  | 0.489  | 0.687  |
| This paper       | 0.354        | 0.566   | 0.603   | 0.371  | 0.505  | 0.669  |

## Reference
* [1] Xuan Dong et al., "Fast efficient algorithm for enhancement of low lighting video," 2011 IEEE International Conference on Multimedia and Expo, Barcelona, 2011, pp. 1-6, doi: 10.1109/ICME.2011.6012107.
* [2] Ge, Z., Liu, S., Wang, F., Li, Z., & Sun, J. (2021). Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430.

</details>

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
@article{yan2022highly,
  title={Highly Differentiated Target Detection Method Based on YOLOv3 Model Under Extremely Low Light Conditions},
  author={Chenxu Yan, Haijian Shao, Xing Deng},
  journal={Available at SSRN 4102943},
  year={2022}
}

@article{shao2024highly,
  title={Highly differentiated target detection under extremely low-light conditions based on improved YOLOx model},
  author={Haijian Shao, Chenxu Yan, Xing Deng, Lihao Qiu, Yingtao Jiang},
  journal={Journal of Electronic Imaging},
  year={2024}
}
```
Feel free to quote articles from my Google Scholar profile: [Google Scholar]: https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en


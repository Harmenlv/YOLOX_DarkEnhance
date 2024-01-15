## Introduction
Python code for our paper (Highly differentiated target detection under extremely low-light conditions based on improved YOLOx model)
Note: The original implementation of PyTorch version YOLOX, please vist: [MegEngine implementation](https://github.com/MegEngine/YOLOX).

## Updates!!
* 【2022/01/20】 We suport jit compile op.[notes]([docs/updates_note.md](https://assets.researchsquare.com/files/rs-1212268/v1_covered.pdf?c=1642711187))
* 【2023/10/21】 We optimize the training process under the low-light condition and obtain higher performance! See [notes](Coming soon) for more details.

## TODO: Update with exact module version
numpy,torch>=1.7,opencv_python,loguru,tqdm,torchvision,thop,ninja,tabulate

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

## 以下安装和开始部分，需要修改

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX from source.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-s -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
or
```shell
python tools/demo.py image -f exps/default/yolox_s.py -c /path/to/your/yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pth --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache]
                               yolox-m
                               yolox-l
                               yolox-x
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8
* --fp16: mixed precision training
* --cache: caching imgs into RAM to accelarate training, which need large system RAM. 

  

When using -f, the above commands are equivalent to:
```shell
python -m yolox.tools.train -f exps/default/yolox_s.py -d 8 -b 64 --fp16 -o [--cache]
                               exps/default/yolox_m.py
                               exps/default/yolox_l.py
                               exps/default/yolox_x.py
```

**Multi Machine Training**

We also support multi-nodes training. Just add the following args:
* --num\_machines: num of your total training nodes
* --machine\_rank: specify the rank of each node

Suppose you want to train YOLOX on 2 machines, and your master machines's IP is 123.123.123.123, use port 12312 and TCP.  
On master machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 0
```
On the second machine, run
```shell
python tools/train.py -n yolox-s -b 128 --dist-url tcp://123.123.123.123:12312 --num_machines 2 --machine_rank 1
```

**Logging to Weights & Biases**

To log metrics, predictions and model checkpoints to [W&B](https://docs.wandb.ai/guides/integrations/other/yolox) use the command line argument `--logger wandb` and use the prefix "wandb-" to specify arguments for initializing the wandb run.

```shell
python tools/train.py -n yolox-s -d 8 -b 64 --fp16 -o [--cache] --logger wandb wandb-project <project name>
                         yolox-m
                         yolox-l
                         yolox-x
```

An example wandb dashboard is available [here](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom0)

**Others**  
See more information with the following command:
```shell
python -m yolox.tools.train --help
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]
                               yolox-m
                               yolox-l
                               yolox-x
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python -m yolox.tools.eval -n  yolox-s -c yolox_s.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
                               yolox-m
                               yolox-l
                               yolox-x
```

</details>
<details>
<summary>Tutorials</summary>

*  [Training on custom data](docs/train_custom_data.md)
*  [Manipulating training image size](docs/manipulate_training_image_size.md)
*  [Freezing model](docs/freeze_module.md)

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
```
## In memory of Dr. Jian Sun
Without the guidance of [Dr. Sun Jian](http://www.jiansun.org/), YOLOX would not have been released and open sourced to the community.


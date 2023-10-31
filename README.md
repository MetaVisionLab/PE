
# Parameter Exchange for Robust Dynamic Domain Generalization
- 🔔This is the official (Pytorch) implementation for the paper "Parameter Exchange for Robust Dynamic Domain Generalization", ACM MM 2023.
- 🛖This repository is built on the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) which is designed for the research of Domain adaptation, Domain generalization, and Domain generalization. You can also view the Dassl project for details: [https://github.com/KaiyangZhou/Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch)

# 🛠️Setup
## Runtime

The main python libraries we use:
- Python 3.8
- torch 1.8.1
- numpy 1.19.2

## Datasets
Please create a directory named `datasets` in current directory, then install these following datasets into `datasets`:
- For [PACS](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#pacs), [VLCS](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#vlcs), [Office-Home](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#office-home-dg), and [DomainNet](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#domainnet) datasets, please view this [site](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#domain-generalization).
- For the Terria Incognita dataset, please view this [site](https://github.com/facebookresearch/DomainBed#quick-start) to manually download and install it. 

You can also change the root directory of datasets by modifying the default value of the argument `--root` in [tools/train.py](tools%2Ftrain.py)[L96]:
```python
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./datasets', help='path to datasets')
```

## Pretrained Weights
Please create a directory named `checkpoints` in current directory, then download following pretrained weights into `checkpoints`: 
- [DDG](https://www.ijcai.org/proceedings/2022/0187): [GoogleDrive](https://drive.google.com/file/d/1U183wQI1O7HP2WydOkpQniw2QdPY5ufw/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Avbw15URUoiwnagTLnW8A1SN1vuRFg?e=lbWOqH), or [Quark](https://pan.quark.cn/s/011448c5c564)
- [DRT](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Dynamic_Transfer_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.html): [GoogleDrive](https://drive.google.com/file/d/12K6jKu-3DzhLf_8pqY5ZwYY3bMPFveaZ/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Avbw15URUoiwnagSvMIq2PB5groJpw?e=QTV1Ii), or [Quark](https://pan.quark.cn/s/db8539f3b814)
- [ODConv](https://openreview.net/pdf?id=DmpCfq6Mg39): [GoogleDrive](https://drive.google.com/file/d/1uhhcBVuI5ZxXBRYFsYGpt7m9zQAigJsV/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Avbw15URUoiwnagUojPiVjEVH72qlQ?e=ov1Ogn), or [Quark](https://pan.quark.cn/s/8dfbb69adcc3)

# 🎢Run
After finishing above steps, your directory structure of code may like this:
```text
DDG_PE/
    |–– checkpoints/
        odconv4x_resnet50.pth.tar
        resnet50_draac_v3_pretrained.pth
        resnet50_draac_v4_pretrained.pth
    |–– configs/
    |–– dataset/
        |–– domainnet/
            |–– clipart/
            |–– infograph/
            |–– painting/
            |–– quickdraw/
            |–– real/
            |–– sketch/
            |–– splits/
        |–– office_home_dg/
            |–– art/
            |–– clipart/
            |–– product/
            |–– real_world/
        |–– terra_incognita/
            |–– location_38/
            |–– location_43/
            |–– location_46/
            |–– location_100/
        |–– VLCS/
            |–– CALTECH/
            |–– LABELME/
            |–– PASCAL/
            |–– SUN/
        |–– paccs/
            |–– images/
            |–– splits/
    |–– dassl/
    |–– tools/
    main.py
    parse_test_res.py
    README.md
    share.py
    train.sh
```
To run the experiment of `DDG w/ CI-PE`, just enter the following cmd on root directory:
```shell
bash train.sh DDG CI PACS
```
Usage of `train.sh`:
```text
bash train.sh {arg1=dymodel} {arg2=pe_type} {arg3=dataset}
```
- `dymodel` is the backbone of the dynamic network, available ones are: `DRT`, `DDG`, `ODCONV`
- `pe_type` determines which PE method to use, available ones are: `CI`,`CK`
- `dataset` specifies which dataset to train and test on, available ones are: `PACS`,`OfficeHome`, `PACS`,`VLCS`, `TerriaIncognita`,`DomainNet`

# 📌Citation
If you would like to cite our works, the following bibtex code may be helpful:
```text
@inproceedings{lin2023pe,
    title={Parameter Exchange for Robust Dynamic Domain Generalization},
    author={Lin, Luojun and Shen, Zhifeng and Sun, Zhishu and Yu, Yuanlong and Zhang, Lei and Chen, Weijie},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
    year={2023},
}

@inproceedings{sun2022ddg,
  title={Dynamic Domain Generalization},
  author={Sun, Zhishu and Shen, Zhifeng and Lin, Luojun and Yu, Yuanlong and Yang, Zhifeng and Yang, Shicai and Chen, Weijie},
  booktitle={IJCAI},
  year={2022}
}
```

# 🔗Acknowledgements
- Our code is built on Dassl.pytorch - https://github.com/KaiyangZhou/Dassl.pytorch
- The Terria Incognita dataset is installed from DomainBed - https://github.com/facebookresearch/DomainBed

# ⚖️License
This source code is released under the MIT license. View it [here](LICENSE)

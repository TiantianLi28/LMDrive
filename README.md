# LMDrive for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [LMDrive](#LMDrive)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
-   [公网地址说明](#公网地址说明)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

LMDrive 是首个将大语言模型运用至自动驾驶端到端、闭环训练上的模型。LMDrive通过将文字指令与图像、雷达信息处理后合并传入大语言模型，最终由大语言模型输出控制信号。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| LMDrive |   Instruction Finetuning   |    ✔     |

## 代码实现

- 参考实现：

    ```
    url=https://github.com/opendilab/LMDrive
    commit_id=43fc2e9a914623fd6eec954a94aeca2d3966e3db
    ```
- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/ModelZoo-PyTorch.git
    code_path=PyTorch/built-in/autonoumous_driving 
    ```

# LMDrive

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.RC3  |
|       CANN        | 8.0.RC3  |
|    昇腾NPU固件    | 24.1.RC3 |
|    昇腾NPU驱动    | 24.1.RC3 |

### 环境配置

- 安装Decord

  - 先安装Decord依赖的ffmpeg包。根据ffmpeg仓的指引，下载ffmpeg压缩包。

    ```
    tar -zxvf ffmpeg-4.2.1.tar.gz
    cd ffmpeg-4.2.1
    ./configure --enable-shared --disable-swresample --disable-x86asm --prefix=/path/to/ffmpeg(安装路径)
    make -j 32（cpu核数）
    make install
    vi ~/.bashrc
    export FFMPEG_PATH=/path/to/ffmpeg/
    export LD_LIBRARY_PATH=$FFMPEG_PATH/lib:$LD_LIBRARY_PATH
    source ~/.bashrc

    yum install -y ffmpeg ffmpeg-devel
    ```

  - 安装Decord

    ```
    git clone --recursive https://github.com/dmlc/decord
    cd decord
    mkdir build && cd build
    cmake ..（可以选择使用cmake .. -DFFMPEG_DIR=/path/to/ffmpeg/）
    make
    cd ../python
    pwd=$PWD
    echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc
    source ~/.bashrc
    python3 setup.py install --user
    ```
- 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

    ```
    conda create -n lmdrive python=3.8
    conda activate lmdrive
    cd vision_encoder
    pip3 install -r requirements.txt
    python setup.py develop # if you have installed timm before, please uninstall it
    cd ../LAVIS
    pip3 install -r requirements.txt
    python setup.py develop # if you have installed LAVIS before, please uninstall it
    ```

- 安装mxDriving加速库，安装master分支，具体方法参考[原仓](https://gitee.com/ascend/mxDriving)。



### 准备数据集

- 根据原仓**Dataset**章节准备数据集，数据集目录及结构如下：

```
LMDrive
├── LAVIS
│   └──dataset
│       └── dataset_index.txt  # for vision encoder pretraining
│       └── navigation_instruction_list.txt  # for instruction finetuning
│       └── notice_instruction_list.json  # for instruction finetuning
│       └── routes_town06_long_w7_11_28_18_28_35  #  data folder
│       └── routes_town01_long_w16_08_13_08_35_38
│       └── routes_town01_long_w18_08_13_07_26_30
│       ├── rgb_full
│       ├── lidar
│       └── ...
```
- 完整数据集大小约2T，若设备内存不足，可仅下载一部分数据集作为测试。训练结果部分使用的数据集在LMDrive/LAVIS/dataset/dataset_used.txt中列出。

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      

### 准备预训练权重

- 根据原仓**LMDrive Weights**章节下载 LMDrive-1.0 (LLaMA-7B) 版本中 VisionEncoder 和 LLM-base 的权重，并在 LMDrive/LAVIS/lavis/projects/lmdrive/notice_llama7b_visual_encoder_r50_seq40.yaml 中将 preception_model_ckpt 和 llm_model 参数的数据分别改为 VisionEncoder 和 LLM 的权重路径。

- 在未联网或设有防火墙的环境中进行训练时，需要将bert-base-uncased模型的checkpoint下载至环境后更改以下文件：
LMDrive/LAVIS/lavis/models/blip2_models/blip2.py：
```
 ln32： 
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
 -> 
   bert_tokenizer_path = 'path/to/bert-base-uncased/'
   tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path, local_files_only=True, truncation_side=truncation_side)
 ln49：
   encoder_config = BertConfig.from_pretrained("bert-base-uncased")
 ->
   bert_tokenizer_path = 'path/to/bert-base-uncased/'
   encoder_config = BertConfig.from_pretrained(bert_tokenizer_path, local_files_only=True)
 ln56：
   Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
 ->
   Qformer = BertLMHeadModel.from_pretrained(bert_tokenizer_path, local_files_only=True, config=encoder_config)
```
LMDrive/LAVIS/lavis/models/drive/blip2.py：
```
 ln32： 
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
 -> 
   bert_tokenizer_path = 'path/to/bert-base-uncased/'
   tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path, local_files_only=True, truncation_side=truncation_side)
 ln49：
   encoder_config = BertConfig.from_pretrained("bert-base-uncased")
 ->
   bert_tokenizer_path = 'path/to/bert-base-uncased/'
   encoder_config = BertConfig.from_pretrained(bert_tokenizer_path, local_files_only=True)
 ln56：
   Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
 ->
   Qformer = BertLMHeadModel.from_pretrained(bert_tokenizer_path, local_files_only=True, config=encoder_config)
```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  1. 在模型根目录下，运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡精度训练

     ```
     bash test/train_8p.sh
     ```

     - 单机8卡性能训练


     ```
     bash test/train_8p_performance.sh
     ```


#### 训练结果
| 芯片          | 卡数 | global batch size | Precision | epoch |  IoU   |  mIoU  | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :----: | :-------------------: |
| 竞品A           |  8p  |         2         |   fp32    |  12   | 0.3163 | 0.1999 |         1028          |
| Atlas 800T A2 |  8p  |         2         |   fp32    |  12   | 0.3114 | 0.1995 |         1054          |

# 公网地址说明

代码涉及公网地址参考 public_address_statement.md

# 变更说明

2024.12.cc 首次发布

# FAQ

无
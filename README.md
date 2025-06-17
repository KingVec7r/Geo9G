### 设置数据和工作路径
```bash
export DATA_BASE_DIR="/data/jr"
export WORKSPACE_DIR="/home/dancer/.cache/jr/workspace"
```

### 数据下载
```bash
# 下载解压 GeoChat_Instruct 数据集
cd "$DATA_BASE_DIR"
git clone https://hf-mirror.com/datasets/MBZUAI/GeoChat_Instruct
cd GeoChat_Instruct
cat images_parta* > images.zip
unzip images.zip

# 下载解压 VRSBench 数据集
cd "$DATA_BASE_DIR"
git clone https://hf-mirror.com/datasets/xiang709/VRSBench
cd VRSBench
for file in *.zip; do unzip "$file" -d "${file%.zip}"; done
```

### 模型权重下载
```bash
# 下载 FM9G4B-V 模型权重
cd "$DATA_BASE_DIR/weights"
wget https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/FM9G4B-V.tar.gz
tar -zxvf FM9G4B-V.tar.gz

# 下载 Qwen2.5-VL-7B-Instruct 模型权重
cd "$DATA_BASE_DIR/weights"
git clone https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct
```

### 新建环境
```bash
cd "$WORKSPACE_DIR"
git clone https://github.com/KingVec7r/Geo9G
cd Geo9G

# geo9g 用于 fm9g4b-v 训练和推理
conda create -n geo9g python=3.10
conda activate geo9g
pip install -r requirements_fm9gv_py10.txt

# vllm 用于 vrsbench 评估
conda create -n vllm python=3.10
conda activate vllm
pip install -r requirements_vllm_py10.txt
```

### 数据处理
```bash
conda activate geo9g

# 将 GeoChat 数据集和 VRSBench 数据集转换为 FM9G4B-V 格式
# 目标框统一格式 <box> x1 y1 x2 y2</box> range from 0 to 1000
python tools/data_process/mix_geochat_vrsbench.py
```

### VRSBench 评估
```bash
# 推理 -m 设置模型路径，默认为 ./FM9G4B-V -t 设置任务类型，默认为 None
python tools/VRSBench/fm9g_infer.py -m FM9G4B-V 

# 评估 -m 设置模型路径，默认为 ./FM9G4B-V -t 设置任务类型，默认为 None -s 指定是否跳过对referring 结果评估的大模型打分阶段，当 skip_llm 为 True 且打分文件存在时跳过
conda activate vllm
python tools/VRSBench/fm9g_eval_on_infer_results.py -m FM9G4B-V -s False

# 绘制 referring 任务的边界框 -m 设置模型路径，默认为 ./FM9G4B-V -n 设置随机采样图像数量，默认为 100
conda activate geo9g
python tools/VRSBench/paint_bbox.py -m FM9G4B-V -n 5
```

### FM9G4B-V 微调（deepspeed zero2 加速）
```bash
cd "$WORKSPACE_DIR/Geo9G"
cd cpm-9g/FM9G4B-V/finetune
conda activate geo9g

MODEL="$WORKSPACE_DIR/Geo9G/FM9G4B-V"
DATA="$DATA_BASE_DIR/geo_mix_vrsb_train_fm9g_format.json"
EVAL_DATA="$DATA_BASE_DIR/geo_mix_vrsb_val_fm9g_format.json"

# finetune_ds.sh 中进行训练配置
sh finetune_ds.sh
```

### 微调后评估
```bash
cd "$WORKSPACE_DIR/Geo9G"

# 推理
conda activate geo9g
python tools/VRSBench/fm9g_infer.py -m cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414

# 评估
conda activate vllm
python tools/VRSBench/fm9g_eval_on_infer_results.py -m cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414

# 绘制边界框
conda activate geo9g
python tools/VRSBench/paint_bbox.py -m cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414
```

如果在模型训练后进行推理时出现以下错误：
```
assert self.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
AttributeError: 'FM9GVTokenizerFast' object has no attribute 'image_processor'
```
建议将FM9G4B-V模型文件夹以下几个文件复制到checkpoint文件夹:
```
image_processing_fm9gv.py
processing_fm9gv.py
preprocessor_config.json
```
注意:根据训练保存的checkpoint文件夹下config.json中的相关参数,修改preprocessor_config.json文件中的参数(max_slice_nums,patch_size等)。
e.g.
```bash
sudo cp FM9G4B-V/image_processing_fm9gv.py cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414
sudo cp FM9G4B-V/processing_fm9gv.py cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414
sudo cp FM9G4B-V/preprocessor_config.json cpm-9g/FM9G4B-V/finetune/output/checkpoint-3414
```
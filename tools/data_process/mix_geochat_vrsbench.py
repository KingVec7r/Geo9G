import json
import re
import os
from tqdm import tqdm
import random

validation_ratio = 0.1

DATA_BASE_DIR = os.environ.get('DATA_BASE_DIR', '/data/jr')

geochat_img_base = os.path.join(DATA_BASE_DIR, "GeoChat_Instruct/share/softwares/kartik/GeoChat_finetuning/final_images_llava")

geochat_train_json = os.path.join(DATA_BASE_DIR, "GeoChat_Instruct/GeoChat_Instruct.json")

vrsbench_img_train_base = os.path.join(DATA_BASE_DIR, "VRSBench/Images_train")

vrsbench_img_val_base = os.path.join(DATA_BASE_DIR, "VRSBench/Images_val")

vrsbench_train_json = os.path.join(DATA_BASE_DIR, "VRSBench/VRSBench_train.json")

result_train_data_file = os.path.join(DATA_BASE_DIR, 'geo_mix_vrsb_train_fm9g_format.json')

result_val_data_file = os.path.join(DATA_BASE_DIR, 'geo_mix_vrsb_val_fm9g_format.json')

def process_string_geo(input_str):
    # 定义正则表达式模式来匹配 {<x1><y1><x2><y2>|<z>} 格式
    pattern = r'\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>\|<(-?\d+)>\}' 
    def replace_match(match):
        # 提取四个坐标值
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        z = int(match.group(5))  # 虽然不用，但还是提取出来
        
        # 验证并调整坐标值
        x1 = max(0, min(100, x1))
        y1 = max(0, min(100, y1))
        x2 = max(0, min(100, x2))
        y2 = max(0, min(100, y2))
        
        # 乘以10并创建新的box标签
        return f'<box>{x1*10} {y1*10} {x2*10} {y2*10}</box>'  
    # 使用正则表达式替换所有匹配项
    return re.sub(pattern, replace_match, input_str)

def process_string_vrsb(input_str):
    # 定义正则表达式模式来匹配 {<x1><y1><x2><y2>|<z>} 格式
    pattern = r'\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>\}' 
    def replace_match(match):
        # 提取四个坐标值
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        
        # 验证并调整坐标值
        x1 = max(0, min(100, x1))
        y1 = max(0, min(100, y1))
        x2 = max(0, min(100, x2))
        y2 = max(0, min(100, y2))
        
        # 乘以10并创建新的box标签
        return f'<box>{x1*10} {y1*10} {x2*10} {y2*10}</box>'  
    # 使用正则表达式替换所有匹配项
    return re.sub(pattern, replace_match, input_str)

def convert_json_structure(json_data, dataset = 'geochat'):
    # 确保输入是字典类型
    if not isinstance(json_data, dict):
        return json_data
    
    converted_data = {}
    
    for key, value in json_data.items():
        if key == "from":
            key = "role"
        elif key == "value":
            key = "content"

        if value == 'human':
            value = 'user'
        elif value == 'gpt':
            value = 'assistant'

        if dataset =='geochat':
            value = process_string_geo(value)   
        elif dataset == 'vrsbench':
            value = process_string_vrsb(value)
        
        converted_data[key] = value
    
    return converted_data

data_for_fm9gv = []

# geochat
try:
    # geochat
    # 读取输入JSON文件
    with open(geochat_train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # data = data[:2]
    # 转换数据格式
    print('Transfer geochat data ...')
    for item in tqdm(data):
        converted_item = {
            "id": item.get('id'),
            "image": os.path.join(geochat_img_base, item.get('image')),
            "conversations": [convert_json_structure(d) for d in item.get('conversations')],
        }
        data_for_fm9gv.append(converted_item)

    # vrsbench
    with open(vrsbench_train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # data = data[:2]
    # 转换数据格式
    print('Transfer vrsbench data ...')
    for item in tqdm(data):
        converted_item = {
            "id": item.get('id'),
            "image": os.path.join(vrsbench_img_train_base, item.get('image')),
            "conversations": [convert_json_structure(d, 'vrsbench') for d in item.get('conversations')],
        }
        data_for_fm9gv.append(converted_item)

        # 随机打乱列表
    random.shuffle(data_for_fm9gv)
    
    # 计算验证集的大小
    val_size = int(len(data) * validation_ratio)
    
    # 划分训练集和验证集
    train_set = data_for_fm9gv[val_size:]
    val_set = data_for_fm9gv[:val_size]
    
    # 写入输出JSON文件
    with open(result_train_data_file, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(result_val_data_file, 'w', encoding='utf-8') as f:
        json.dump(val_set, f, ensure_ascii=False, indent=2)
    
    print(f"成功转换并保存到 {result_train_data_file}")

except Exception as e:
    print(f"error: {e}")

import os
import json
import torch
import re
import logging
import argparse

from sacrebleu.metrics import BLEU
from statistics import mean
from torchvision.ops import box_iou
      
class VRSBenchEval:
    def __init__(self, judge_model_path, model_file_path, task=None, skip_llm=False):
        parts = model_file_path.rstrip('/').split('/')
        model_name = parts[-1] if parts else ''

        self.judge_model_path = judge_model_path

        infer_results_path = f"./tools/VRSBench/eval_result_{model_name}"

        self.results_file_path = os.path.join(infer_results_path, f"VRSBench_eval_result_{task}_{model_name}.json")
        self.results_file_dict = {t:os.path.join(infer_results_path, f"VRSBench_eval_result_{t}_{model_name}.json") for t in ['cap','referring','vqa']}

        self.task = task
        self.skip_llm = skip_llm
        
        self.llm_judeg_results_file_path = os.path.join(infer_results_path, f"VRSBench_vqa_llm_judge_result_{model_name}.json")

        log_file_name = os.path.join(infer_results_path, "eval.log")

        logging.basicConfig(
            level=logging.INFO,  # 日志级别：DEBUG < INFO < WARNING < ERROR < CRITICAL
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
            filename=log_file_name  # 输出到文件（可选）
        )
        # 获取logger对象
        self.logger = logging.getLogger(__name__)
          
    def run(self):
        self.vrsbench_eval_ddp()

    def vrsbench_eval_ddp(self):
        if self.task:
            infer_results = self.load_json_file(self.results_file_path)
            self.result_eval(infer_results)
        else:
            for t in ['cap','referring','vqa']:
                self.task = t
                self.results_file_path = self.results_file_dict[t]
                infer_results = self.load_json_file(self.results_file_path)
                self.result_eval(infer_results)
        
    def result_eval(self, infer_results):    
        if self.task == "cap":
            # 计算BLEU分数
            print("Calculating BLEU score...")
            bleu_dict = {
                "bleu_1": [],
                "bleu_2": [],
                "bleu_3": [],
                "bleu_4": [],
                "hyp_len": []
            }
            bleu = BLEU()
            for item in infer_results:
                bleu_score = bleu.corpus_score([item['inference_result']], [[item['ground_truth']]])

                bleu_dict["bleu_1"].append(bleu_score.precisions[0])
                bleu_dict["bleu_2"].append(bleu_score.precisions[1])
                bleu_dict["bleu_3"].append(bleu_score.precisions[2])
                bleu_dict["bleu_4"].append(bleu_score.precisions[3])
                bleu_dict["hyp_len"].append(bleu_score.sys_len)
            
            avg_bleu_dict = {k:mean(v) for k,v in bleu_dict.items()}
           
            print(f"BLEU-1: {avg_bleu_dict['bleu_1']:.2f}")
            print(f"BLEU-2: {avg_bleu_dict['bleu_2']:.2f}")
            print(f"BLEU-3: {avg_bleu_dict['bleu_3']:.2f}")
            print(f"BLEU-4: {avg_bleu_dict['bleu_4']:.2f}")
            print(f"Avg_L: {avg_bleu_dict['hyp_len']:.2f}")

        elif self.task == "referring":
            # 计算IoU分数
            bbox_infer = [self.parse_bbox_infer(res) for res in [r['inference_result'] for r in infer_results]]
            bbox_gt = [self.parse_bbox_gt(gt) for gt in [r['ground_truth'] for r in infer_results]]

            boxes_gt_valid = []
            boxes_infer_valid = []
            for bbox_i, bbox_g in zip(bbox_infer, bbox_gt):
                if isinstance(bbox_i, list) and isinstance(bbox_g, list):
                    boxes_gt_valid.append(bbox_g)
                    boxes_infer_valid.append(bbox_i)
            boxes_a = torch.tensor(boxes_gt_valid) * 10  # GT框 0-100 -> 0-1000
            boxes_b = torch.tensor(boxes_infer_valid)
            iou = box_iou(boxes_a, boxes_b)
            iou = iou.diag().tolist()
            # 计算有效IoU率
            valid_iou_rate = len(iou) / len(bbox_infer)
            # 计算有效IoU率
            iou_avg = sum(iou) / len(iou)
            # 计算IoU@0.5
            iou_at_05 = sum(1 for i in iou if i > 0.5) / len(iou)
            # 计算IoU@0.7
            iou_at_07 = sum(1 for i in iou if i > 0.7) / len(iou) 
            # 输出结果
            print(f"有效IoU率: {valid_iou_rate:.2%}")
            print(f"平均IoU: {iou_avg:.2f}")
            print(f"Acc@0.5: {iou_at_05:.2%}")
            print(f"Acc@0.7: {iou_at_07:.2%}")

        elif self.task == "vqa": # vllm 大模型打分
            # infer_results = infer_results[:32]

            type_list = ['Category', 'Presence', 'Quantity', 'Color', 'Shape', 'Size','Position','Direction','Scene','Reasoning']

            if os.path.exists(self.llm_judeg_results_file_path) and self.skip_llm:
                # 已有llm评分结果
                llm_judge_results = self.load_json_file(self.llm_judeg_results_file_path)
            else:
                question_list = [r['question'] for r in infer_results]
                ground_truth_list = [r['ground_truth'] for r in infer_results]
                inference_result_list = [r['inference_result'] for r in infer_results]
                
                # 生成提示词
                prompts_for_judge = [f"""Question: {q}, Ground Truth Answer: {gt}, Predicted Answer: {pred}. Does the predicted answer match the ground truth? Answer with 1 for match and 0 for no match. ONLY output 0 or 1 —no analysis, explanations, or extra text. Synonyms (e.g., "pond" and "swimming pool") count as matches.""" for q, gt, pred in zip(
                    question_list, ground_truth_list, inference_result_list
                )]

                prompts_for_type = [f"""'Question: {q}, Answer: {gt}'. Select the most appropriate tag for the above QA pair. The tag should be chosen from the candidate list and should reflect the most prominent attribute or aspect that the Q&A focuses on. Your response should include only the tag word —no explanations, punctuation, or additional text. Candidate tags: {str(type_list)}""" for q, gt in zip(
                    question_list, ground_truth_list
                )]
                
                print('Generating llm judge results...')
                prompts = prompts_for_judge + prompts_for_type
                preds = self.llm_generate(prompts)

                llm_judges = preds[:len(preds)//2]
                llm_types = preds[len(preds)//2:]

                judeg_results = [[self.extract_first_number(lj) for lj in lj_i] for lj_i in llm_judges]
                type_results = [[self.extract_first_word(lt, type_list) for lt in lt_i] for lt_i in llm_types]

                final_judge_results = self.extract_majority(judeg_results)
                final_type_results = self.extract_majority(type_results)

                llm_judge_results = [
                    {
                        **r,
                        'llm_judge':lj,
                        'llm_type':lt,
                        'llm_type_ori':lt_o,
                        'llm_judge_result':fj,
                        'llm_type_result':ft
                    } for r, lj, lt, lt_o, fj, ft in zip(
                        infer_results, 
                        judeg_results, 
                        type_results,
                        llm_types,
                        final_judge_results,
                        final_type_results
                    )
                ]
        
                # 保存结果
                with open(self.llm_judeg_results_file_path, 'w', encoding='utf-8') as f:
                    json.dump(llm_judge_results, f, ensure_ascii=False, indent=4)
                self.logger.info(f"✅ LLM 评估结果已保存至 {self.llm_judeg_results_file_path}")

            count_dict = {k:[] for k in type_list}
            valid_judge = 0
            valid_category = 0
            vqa_right_count = 0
            for item in llm_judge_results:
                if item.get('llm_judge_result') in ['0', '1']:
                    item['llm_judge_result'] = int(item['llm_judge_result'])
                    valid_judge += 1
                    vqa_right_count += item['llm_judge_result']

                if item.get('llm_type_result') in type_list:
                    count_dict[item.get('llm_type_result')].append(item.get('llm_judge_result'))
                    valid_category += 1

            eval_Indicator = {
                'LLM分类有效率': valid_category / len(llm_judge_results),
                'LLM评分有效率': valid_judge / len(llm_judge_results),
                'vqa总准确率': vqa_right_count / valid_judge,
                **{ t:sum(count_dict[t]) / len(count_dict[t]) if len(count_dict[t]) else 0 for t in type_list}
            }

            for key, value in eval_Indicator.items(): 
                print(f"{key}: {value:.2%}")
           
    def parse_bbox_infer(self, bbox_str):
        # 匹配模式: 数字序列，可能包含小数点
        pattern = r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>'
        match = re.search(pattern, bbox_str)
        try:
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                bbox =  [x1, y1, x2, y2]
                # 检查坐标是否在有效范围内
                if all(0 <= coord <= 1000 for coord in bbox):
                    return bbox
                else:
                    self.logger.info(f"Bounding box coordinates out of range (infer): {bbox_str}")
                    return 0
            else:
                self.logger.error(f"Error parsing bounding box (infer): {bbox_str}")
                return 1
        except Exception as e:
            self.logger.error(f"Unexpected error parsing bounding box (infer): {bbox_str}, Error: {e}")
            return 2
        
    def parse_bbox_gt(self, bbox_str):
        # 匹配模式: 数字序列，可能包含小数点
        pattern = r'<(\d+)><(\d+)><(\d+)><(\d+)>'
        match = re.search(pattern, bbox_str)
        try:
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                bbox =  [x1, y1, x2, y2]
                # 检查坐标是否在有效范围内
                if all(0 <= coord <= 100 for coord in bbox):
                    return bbox
                else:
                    self.logger.info(f"Bounding box coordinates out of range (gt): {bbox_str}")
                    return 0
            else:
                self.logger.error(f"Error parsing bounding box (gt): {bbox_str}")
                return 1
        except Exception as e:
            self.logger.error(f"Unexpected error parsing gt bounding box (gt): {bbox_str}, Error: {e}")
            return 2
        
    def load_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # 验证数据格式是否为列表
                if not isinstance(data, list):
                    raise ValueError("JSON文件内容不是列表格式")
                # 验证列表中的元素是否为字典
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError("列表中的元素不是字典格式")
                return data
        except FileNotFoundError:
            print(f"错误: 文件 '{file_path}' 不存在")
            return []
        except json.JSONDecodeError:
            print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
            return []
        except ValueError as ve:
            print(f"错误: {ve}")
            return []
        except Exception as e:
            print(f"发生未知错误: {e}")
            return []
        
    def llm_generate(self, prompts):
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(
            temperature=0.45, 
            top_p=0.9, 
            max_tokens=64,
            n=5
            )
        llm = LLM(
            model=self.judge_model_path,
            dtype='float16',
            # gpu_memory_utilization=0.5,
            tensor_parallel_size=torch.cuda.device_count())
        outputs = llm.generate(prompts, sampling_params)
        res_list = []
        for output in outputs:
            # n = 5
            all_responses = [o.text.strip() for o in output.outputs]
            res_list.append(all_responses)
            #  n = 1
            # response_text = output.outputs[0].text.strip()
            # res_list.append(response_text)        
        return res_list
    
    def extract_first_number(self, text):
        """提取字符串中的第一个数字"""
        pattern = r'\d+'  # 匹配一个或多个数字
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None

    def extract_first_word(self, text, type_list):
        """提取字符串中的第一个单词"""
        # pattern = r'(?:Tag:)?\s*([A-Z][a-z]*)'  # 匹配一个或多个字母组成的单词
        pattern = r'[A-Z][a-z]+'
        matches = re.findall(pattern, text)
        
        # 如果匹配到单词且第一个是"Tag"，则返回第二个
        if matches and matches[0] == "Tag" and len(matches) > 1:
            Tag = matches[1]
        elif matches:
            Tag = matches[0]
        if Tag in type_list:
            return Tag
        return None

    def extract_majority(self, pred_list):
        """从每个子列表中提取出现至少两次的元素，如果没有则返回None"""
        result = []
        
        for sublist in pred_list:
            # 统计每个元素出现的次数
            count_dict = {}
            for element in sublist:
                count_dict[element] = count_dict.get(element, 0) + 1
            
            # 查找出现至少两次的元素
            majority_element = None
            max_count = 0
            
            for element, count in count_dict.items():
                if count > max_count:
                    max_count = count
                    majority_element = element

            result.append(majority_element)
        
        return result
    
    def create_directory_safe(self, path: str) -> None:
        """
        安全创建目录的函数，包含错误处理
        :param path: 目录路径
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"目录 {path} 创建成功")
            else:
                print(f"目录 {path} 已存在")
        except PermissionError:
            print(f"错误：没有权限创建目录 {path}")
        except FileExistsError:
            print(f"错误：路径 {path} 已存在且不是目录")
        except Exception as e:
            print(f"错误：创建目录 {path} 时发生未知错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型推理脚本')
    parser.add_argument('-m', '--model_path', 
                        type=str, 
                        default='./FM9G4B-V',
                        help='模型文件路径')
    parser.add_argument('-s', '--skip_llm', 
                        type=str, 
                        default=False,
                        help='已有llm评估文件时可以跳过大模型评估直接计算准确率（referring）')
    parser.add_argument('-t', '--task', 
                        type=str, 
                        default=None,
                        help='任务类型')
    args = parser.parse_args()

    # 模型文件路径
    model_file_path = args.model_path
    task = args.task
    skip_llm = args.skip_llm.lower() == "true"
    print(skip_llm)
    DATA_BASE_DIR = os.environ.get('DATA_BASE_DIR', '/data/jr')
    judge_model_path = os.path.join(DATA_BASE_DIR, "weights/Qwen/Qwen2.5-7B-Instruct")
    
    eval = VRSBenchEval(
        judge_model_path = judge_model_path,
        model_file_path = model_file_path, 
        task = task, # 任务类型：cap, referring, vqa， default = 遍历所有任务
        skip_llm =skip_llm
        )

    eval.run()
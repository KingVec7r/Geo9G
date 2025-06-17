# 九格通用基础大模型
## 简介
启元九格大模型由启元实验室牵头，联合清华大学、哈尔滨工业大学、中国科学院计算技术研究所、北京大学、南开大学等优势单位共同研制。具有高效训练与推理和高效适配与部署的技术特点，具备文本问答、文本分类、机器翻译、文本摘要等自然语言处理能力。

## 更新信息
- 本次启元九格开源4B多模态基础大模型，具体的模型推理等内容见：[QUICK START](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM9G-V/quick_start_clean/readmes/quick_start_9g4bv.md)
- 本次启元九格开源13B多模态基础大模型，具体的模型推理等内容见：[QUICK START](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM9G-V/quick_start_clean/readmes/quick_start.md)
- 若还在使用旧版本的九格模型训练和推理，请切换分支到[master](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/master/quick_start_clean/readmes/README_ALL.md)

<!-- ## 版本更新内容
  具体的迭代信息如下：                                                                               
- 训练：升级了训练代码，提升GPU利用率和并行化，并且2B模型能兼容transformers中的tokenizer(LlamaTokenizerFast)
- 推理：支持vllm进行模型推理和部署，可以接入langchain、openai等部署方式；同时可以支持2b模型转换成GGUF等多种部署格式的部署
- 由于新架构中多数据集验证发现2B模型进行lora训练效果不及全参数微调，因此建议2B模型全参数微调，8B模型LORA微调在master分支进行                              
                                    
## 2024.08.19 NOTICE
- 由于新架构中多数据集验证发现2B模型进行lora训练效果不及全参数微调
- 2B模型采用全参数微调训练，我们在[QUICK START](https://www.osredm.com/jiuyuan/CPM-9G-8B/tree/FM_9G/quick_start_clean/readmes/quick_start.md) 中更新了更多关于微调训练的信息
- 8B模型LORA微调在master分支进行训练                      -->

# 迈向通用智能的大模型技术系列课程
系列课程全方位介绍人工智能和大模型技术的基础知识和前沿课题，理论学习和实践应用相结合。课程既有“人工智能与大模型通论”和“神经网络与预训练模型”等基础知识，也有“九格大模型生态体系”和“领域大模型实战”等实战主题，基本内容包括大模型训练、微调、知识增强、伦理安全、多模态、具身智能、自主智能体等话题，高级选题包括多语言处理、面向科学研究的大模型应用、高效计算技术、评测与数据科学等话题。课程旨在通过一系列精心设计的单元为学习者提供大型通用人工智能的学习之旅。

## 人工智能大模型通论
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%80%9A%E8%AE%BA-%E5%AD%99%E8%8C%82%E6%9D%BE%E8%80%81%E5%B8%88-1124_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[人工智能与大模型通论-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/1.%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%80%9A%E8%AE%BA-PPT.pdf)
                                                      
## 大模型技术的重要特性与发展趋势
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8A%80%E6%9C%AF%E7%9A%84%E9%87%8D%E8%A6%81%E7%89%B9%E6%80%A7%E4%B8%8E%E5%8F%91%E5%B1%95%E8%B6%8B%E5%8A%BF-%E5%88%98%E7%9F%A5%E8%BF%9C%E8%80%81%E5%B8%88-1201_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型技术的重要特性与发展趋势-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%8A%80%E6%9C%AF%E7%9A%84%E9%87%8D%E8%A6%81%E7%89%B9%E6%80%A7%E4%B8%8E%E5%8F%91%E5%B1%95%E8%B6%8B%E5%8A%BF-PPT.pdf)
                 
## 大语言模型的适配与对齐技术
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2023-12-22-%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%80%82%E9%85%8D%E4%B8%8E%E5%AF%B9%E9%BD%90%E6%8A%80%E6%9C%AF-%E4%B8%81%E5%AE%81_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大语言模型的适配与对齐技术-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/3.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E9%80%82%E9%85%8D%E4%B8%8E%E5%AF%B9%E9%BD%90%E6%8A%80%E6%9C%AF-PPT.pdf)
                  
## 大模型领域适配原理与实战
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/2023-12-29%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%86%E5%9F%9F%E9%80%82%E9%85%8D%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E6%88%98-%E7%8E%8B%E7%A1%95_DeWatermark.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型领域适配原理与实战-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/4.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E9%A2%86%E5%9F%9F%E9%80%82%E9%85%8D%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E6%88%98-PPT.pdf)
                
## 知识增强的大语言模型
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E7%9F%A5%E8%AF%86%E5%A2%9E%E5%BC%BA%E7%9A%84%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.mp4
" width="800px" height="600px" controls="controls"></video>
[知识增强的大语言模型-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/5.%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%9A%84%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B-PPT.pdf)
                   
## 大模型工具学习
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B7%A5%E5%85%B7%E5%AD%A6%E4%B9%A0.mp4
" width="800px" height="600px" controls="controls"></video>
[大模型工具学习-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/6.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%B7%A5%E5%85%B7%E5%AD%A6%E4%B9%A0-PPT.pdf)
                 
## 检索增强生成的基本实现
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%AE%9E%E7%8E%B0.mp4
" width="800px" height="600px" controls="controls"></video>
[检索增强生成的基本实现-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/7.%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%AE%9E%E7%8E%B0-PPT.pdf)
              
## 多模态语义检索与检索增强技术
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%AD%E4%B9%89%E6%A3%80%E7%B4%A2%E4%B8%8E%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF.mp4
" width="800px" height="600px" controls="controls"></video>
[多模态语义检索与检索增强技术-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/8.%E5%A4%9A%E6%A8%A1%E6%80%81%E8%AF%AD%E4%B9%89%E6%A3%80%E7%B4%A2%E4%B8%8E%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E6%8A%80%E6%9C%AF-PPT.pdf)
              
## 大语言模型驱动的多智能体协作与演化
<video src="https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/400_0121.mp4
" width="800px" height="600px" controls="controls"></video>
[大语言模型驱动的多智能体协作与演化-PPT](https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/%E8%AF%BE%E7%A8%8B%E8%A7%86%E9%A2%91/9.%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93%E5%8D%8F%E4%BD%9C%E4%B8%8E%E6%BC%94%E5%8C%96-PPT.pdf)
                        
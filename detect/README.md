# Paper Error Detection

目录下提供的脚本实现以下功能
1. 从openreview获取文章，并过滤掉过长的文章。（已优化完毕）

    1_0_download_openreview: 从Openreview下载文章，目前支持ICML ICLR NeurIPS, 其余如CVPR ACL等由于论文并不托管在Openreview上，故不支持。

    1_1_filter_overlong_pdf：把太长的文章过滤掉，防止解析有问题。


2. 把pdf格式的文章解析成支持MLLM输入的json格式。（已优化完毕）

    2_0_parse_paper_multi：用Llama-Parse解析文章，并用GPT5进行OCR确保公式等解析的准确性，最后图文并列组织成文章的初始格式，命名为paper_parse_origin.json.

    2_1_add_section：把文章切分成比较详细的blocks，其中每个blocks都标注上对应的Section，例如Abstract等，处理完后的文章命名为paper_parse_add_section.json

    2_2_filter_parse：用规则过滤掉文章的页眉页脚，然后把Checklist部分给删掉，处理完后得到最终的用于后续处理的文章，命名为paper_final.json


3. 用大模型端到端篡改数据。（已优化完毕）

    3_0_synth_corruptions_for_detector_multi：设置模型和并发数，然后就根据Prompt里面写的八类错误修改论文。 同时加了一个参数 --overwrite_apply， 这个设置一个int，目的是对applied小于等于这个数的json重新运行一遍。

    这里采用的模型有8类，5类是国外模型（用自己的4z api调），3类是国内模型（借用xcd的api调）。分别是：gpt-5-2025-08-07，o4-mini，gemini-2.5-pro，claude-sonnet-4-5-20250929，grok-4；qwen3-vl-235b-a22b-instruct，doubao-seed-1-6-251015，glm-4.6。

    其中，gpt-5-2025-08-07 和 o4-mini 是最省心的，基本上一遍过； gemini-2.5-pro，claude-sonnet-4-5-20250929，grok-4，doubao-seed-1-6-251015 这几个一遍基本上都有几个是没法提出json的，多试试几遍也可以过。 Qwen3-VL-235B-A22B-Instruct 则有api调用速度限制，经常报 tpm limit，但是正常得到的结果一般解析出来也没问题；而glm-4.6经常报 "API 调用参数有误"，感觉在这边国产模型的接口还不是很好。

    最后，对NeurIPS_35的数据集进行了所有模型的编辑，因为这边是想做一个表，就是不同模型检测不同模型的错误的表（例如上述就是 8*8=64）的表，而不是全量做。 对ICLR和ICML的数据集用 gpt-5-2025-08-07 和 o4-mini 标注了一遍。 
    
    此外在标的时候还把那些图片太多的解析都删掉了（这里大概又删了十几篇），总之，最后剩下 ICLR 107篇， ICML 67篇， NeurIPS 46篇。

    最后存下的文件为 paper_synth_{MODEL_NAME}.json， 最顶层有两个key为 "edits"、 "apply_results" 和 "paper"， "edits" 里重点记录了篡改的错误原文和替换文，以及错误类型和错误解释，"needs_cross_section" 表示这个错误需不需要通过跨章节检测得出（因为如果是在小模型对Section的切片检测里面，needs_cross_section的true的错误是检测不出来的； "apply_results" 里的 "applied" 这个key 显示了错误是否被加入篡改后的文章 (因为我的逻辑是让LLM输出篡改的原文和篡改后的内容，然后根据字符匹配把篡改后的内容丢进去，如果有时候他篡改表格、公式什么的和原文不完全对的上就无法篡改成功)；最后的"paper" 这个key里面就是完整记录了篡改后的原文。


4. Agent框架端到端检测。（未优化）

    4_0_single_error_detection.py：单智能体模式。

    4_1_mas_error_detection.py：多智能体模式。


5. 端到端评估。（未优化）

    5_eval_detection.py



## 关于RL的做法

一个就是拿已经合成好的这些数据，例如gpt5和o4-mini的合成数据，里面的error_explanation可以喂给reward model作为ground-truth。对于小模型来说输入全文肯定是不太现实的，可以考虑把我的这些合成数据以Section切片成一些小片，那些"needs_cross_section": true的错误就可以不考虑了（因为需要其他章节的证据检测）。

SFT： 目的主要是让LLM输出符合的格式，这个格式需要自己想和构建，可以参考我这样输出json，但是内容不要太多，一般就是罗列出一些疑问点，如"target"（这里不一定要给出精确的位置了，因为不涉及到替换） "error_explanation" （就是解释错误是什么） "suggest_fix" （建议的修改意见）。

RL： 目的是让模型输出精准又具有批判性的意见。 这里的Reward可以参考 我5_eval_detection 的用法，就是拿一个外部的LLM （或者本地配一个72B的，不然外部调太慢，而且太贵了），具体应该有两个作用： 1是给ground-truth 判断 模型的输出有多少个匹配上了ground-truth，即precision；但是如果只有这样，模型肯定就疯狂输出越多越好，对于训练和实际使用都是不利的，因此还需要考虑是不是模型是不是”多虑“，这部分可以通过规则让reward短，或者用LLM再检测一下输出部分是不是包含了很多没什么用的，从而减小这部分的风险。
### TRT-LLM Qwen-7B

本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目，具体选题内容为 2+4 ，即用TensorRT-LLM实现新模型，并在本模型上启用现有feature或添加新feature。

本工作选用[Qwen-7B](https://github.com/QwenLM/Qwen-7B)模型作为待实现的新模型，通义千问-7B（Qwen-7B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。Qwen-7B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到，在Qwen-7B的基础上，还使用对齐机制打造了基于大语言模型的AI助手Qwen-7B-Chat。Qwen-7B支持8k的token长度，在多个全面评估自然语言理解与生成、数学运算解题、代码生成等能力的评测数据集上，均超出了同规模大语言模型的表现。

本工作将Qwen-7B模型移植至TensorRT LLM框架中，并开启了Int8和Int4的权重量化，在fp16精度下，相较启用了fast-attention的原版Qwen-7B transformers实现**1.946x**加速，Rouge精度相差在合理范围内；在int8权重量化下，相较启用了fast-attention的原版Qwen-7B transformers实现**2.754x**加速，Rouge精度相差在合理范围内；在int4量化下，实现**2.334x**加速，Rouge精度有一定损失，但速度、内存占用和引擎大小均大幅缩减，详细数据请参见优化效果部分。

具体模型构建和测试复现代码，请参考`tensorrt_llm_july-release-v1/examples/qwen/README.md`完成构建和运行

<!--

### 主要开发工作

#### 开发工作的难点

请在这一节里总结你的工作难点与亮点。
- 如果使用 TensorRT 进行优化，请介绍一下在模型在导出时、或用polygraphy/trtexec解析时，或在使用TensorRT中，遇到了什么问题并解决了。换句话说，针对这个模型，我们为什么需要额外的工程手段。
- 如果使用 TensorRT-LLM 进行优化，描述以下方面可供选手参考：如果搭建了新模型， 请介绍模型结构有无特别之处，在模型的搭建过程中使用了什么算子，有没有通过plugin支持的新算子。如果支持新feature，请介绍这个feature具体需要修改哪些模块才能实现。如果优化已有模型，请介绍模型性能瓶颈以及解决方法。另外还可以包含工程实现以及debug过程中的难点。

### 开发与优化过程

这一部分是报告的主体。请把自己假定为老师，为 TensorRT 或 TensorRT-LLM 的初学者讲述如何从原始模型出发，经过一系列开发步骤，得到优化后的 TensorRT 或 TensorRT-LLM 模型。或者你是如何一步步通过修改哪些模块添加了新feature的。

建议：

- 分步骤讲清楚开发过程
- 最好能介绍为什么需要某个特别步骤，通过这个特别步骤解决了什么问题
  - 比如，通过Nsight Systems绘制timeline做了性能分析，发现attention时间占比高且有优化空间（贴图展示分析过程），所以决定要写plugin。然后介绍plugin的设计与实现，并在timeline上显示attention这一部分的性能改进。

-->

### 优化效果

这一部分介绍你的工作在云主机上的运行效果。如果是优化模型，需要分两部分说明：

#### 实验环境

#### 性能及精度指标

在CNN Dailymail数据集上完成Summarize任务并使用 [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) 来对比模型优化前后的精度差距。

hf原版实现CNN Dailymail数据集上的Summary任务指标

```
[09/19/2023-16:28:24] [TRT-LLM] [I] Hugging Face (total latency: 63.0036199092865 sec)
[09/19/2023-16:28:24] [TRT-LLM] [I] HF beam 0 result
[09/19/2023-16:28:24] [TRT-LLM] [I]   rouge1 : 23.542276175618408
[09/19/2023-16:28:24] [TRT-LLM] [I]   rouge2 : 8.298578499008089
[09/19/2023-16:28:24] [TRT-LLM] [I]   rougeL : 18.020127809419513
[09/19/2023-16:28:24] [TRT-LLM] [I]   rougeLsum : 19.067740468195073
```

trt llm在CNN Dailymail数据集上的Summary任务指标，采用FP16精度，对应加速比为1.946x，精度误差在合理范围内，GPU显存占用20.249GB，引擎大小16.0GB

```
[09/19/2023-16:35:33] [TRT-LLM] [I] TensorRT-LLM (total latency: 32.374117851257324 sec)
[09/19/2023-16:35:33] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/19/2023-16:35:33] [TRT-LLM] [I]   rouge1 : 25.43300079943117
[09/19/2023-16:35:33] [TRT-LLM] [I]   rouge2 : 9.06443649019336
[09/19/2023-16:35:33] [TRT-LLM] [I]   rougeL : 19.480772591016652
[09/19/2023-16:35:33] [TRT-LLM] [I]   rougeLsum : 20.84898295493978
```

trt llm在CNN Dailymail数据集上的Summary任务指标，采用Int8 WeightOnly量化，对应加速比为2.754x，精度误差在合理范围内，运行时显存占用13.557GB，引擎大小8.4GB

```
[09/20/2023-06:44:54] [TRT-LLM] [I] TensorRT-LLM (total latency: 22.8713800907135 sec)
[09/20/2023-06:44:54] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-06:44:55] [TRT-LLM] [I]   rouge1 : 23.90645420218702
[09/20/2023-06:44:55] [TRT-LLM] [I]   rouge2 : 7.891516095599542
[09/20/2023-06:44:55] [TRT-LLM] [I]   rougeL : 17.593275690274655
[09/20/2023-06:44:55] [TRT-LLM] [I]   rougeLsum : 19.91946017932637
```

trt llm在CNN Dailymail数据集上的Summary任务指标，采用Int4 WeightOnly量化，对应加速比为，精度存在一定误差，运行时显存占用10.542GB，引擎大小5.4GB

```
[09/20/2023-07:02:53] [TRT-LLM] [I] TensorRT-LLM (total latency: 26.977251052856445 sec)
[09/20/2023-07:02:53] [TRT-LLM] [I] TensorRT-LLM beam 0 result
[09/20/2023-07:02:53] [TRT-LLM] [I]   rouge1 : 23.70251298379219
[09/20/2023-07:02:53] [TRT-LLM] [I]   rouge2 : 7.016004368646156
[09/20/2023-07:02:53] [TRT-LLM] [I]   rougeL : 18.027980286289
[09/20/2023-07:02:53] [TRT-LLM] [I]   rougeLsum : 21.013849142346537
```

相关测试代码均包含在本仓库中。

<!--
### Bug报告（可选）

提交bug是对TensorRT/TensorRT-LLM的另一种贡献。发现的TensorRT/TensorRT-LLM或cookbook、或文档和教程相关bug，请提交到[github issues](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues)，并请在这里给出链接。  

对于每个bug，请标记上hackathon2023标签，并写好正文：

- 对于cookbook或文档和教程相关bug，说清楚问题即可，不必很详细。
- 对于TensorRT bug，首先确认在云主机上使用NGC docker + TensorRT 9.0.0.1可复现。
- 然后填写如下模板，并请导师复核确认（前面“评分标准”已经提到，确认有效可得附加分）：
  - Environment
    - TensorRT 9.0.0.1
    - Versions of CUDA, CUBLAS, CuDNN used
    - Container used
    - NVIDIA driver version
  - Reproduction Steps
    - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
  - Expected Behavior
    - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
  - Actual Behavior
    - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
  - Additional Notes
    - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

### 送分题答案（可选）

如果你做了送分题，请把答案写在这里。

### 经验与体会（可选）

欢迎在这里总结经验，抒发感慨。
-->

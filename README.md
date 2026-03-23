# nanoGPT
基于 nanoGPT 实现的轻量化中文唐诗生成模型，支持诗词自动生成、开头续写、本地 CPU/GPU 快速训练与推理。

## 项目背景
nanoGPT 是 Andrej Karpathy 开源的轻量级 GPT 实现，本项目基于该框架做了以下适配优化：
- 适配中文唐诗语料的字符级编码
- 简化训练配置，降低硬件门槛
- 优化生成逻辑，提升诗词通顺度
- 提供完整的「数据预处理→训练→生成」全流程

 ## 项⽬依赖：
| 库名         | 版本要求 | 功能说明               |
| ------------ | -------- | ---------------------- |
| pytorch      | < 3.0    | 模型训练与推理核心框架 |
| numpy        | < 3.0    | 数值计算基础           |
| transformers | < 3.0    | 辅助数据处理           |
| datasets     | < 3.0    | 加载训练数据集         |
| tiktoken     | < 3.0    | 实现 BPE 编码          |
| tqdm         | < 3.0    | 显示代码运行进度条     |

#### 1.1 克隆仓库
git clone https://github.com/zhzh520/DMX_nanoGPT.git
cd nanoGPT-master

#### 1.2 安装依赖

```
pip install torch<3.0 numpy<3.0 transformers<3.0 datasets<3.0 tiktoken<3.0 tqdm<3.0
```

### 2. 数据预处理
#### 2.1 准备语料
项目已内置 tang_poet.txt（唐诗语料）
文本格式要求：每行一首诗，编码为 UTF-8

#### 2.2 执行预处理
将文本转为模型可读取的二进制文件：

python prepare.py

```
import os
import requests  
import tiktoken
import numpy as np

input_file_path = 'tang_poet.txt'

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"错误：未找到文件 {input_file_path}，请检查文件路径是否正确")

with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()

data = data.replace('\u3000', '').replace('\xa0', '').strip()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__),'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__),'val.bin'))

print("✅ 数据预处理完成！已生成 train.bin 和 val.bin")
```

执行后生成：

| 文件      | 占比 / 作用        | 备注                          |
| --------- | ------------------ | ----------------------------- |
| train.bin | 90% - 模型训练数据 | 二进制格式，加载速度更快      |
| val.bin   | 10% - 模型验证数据 | 用于监控过拟合                |
| meta.pkl  | - 字符映射表       | 记录「汉字→数字索引」对应关系 |

### 3. 训练配置
创建 / 修改 `config/train_poemtext_char.py`
代码如下：

```
# ===================== 输出与日志配置 =====================
out_dir = 'out-poemtext-char'          # 模型权重输出目录
eval_interval = 250                    # 每250步验证一次（监控过拟合）
eval_iters = 200                       # 验证集迭代次数（平衡精度与速度）
log_interval = 10                      # 每10步打印一次训练日志
always_save_checkpoint = False         # 仅保存最优模型，减少磁盘占用

# ===================== 数据集配置 =====================
dataset = 'poemtext'                   # 数据集标识
gradient_accumulation_steps = 1        # 梯度累积步数（显存不足可设2/4）
batch_size = 64                        # 批次大小（显存不足改32/16）
block_size = 256                       # 上下文长度（适配唐诗字符数）

# ===================== 模型结构配置 =====================
n_layer = 6                            # Transformer 解码器层数（轻量版）
n_head = 6                             # 多头注意力头数（需满足 n_embd%n_head==0）
n_embd = 384                           # 嵌入向量维度（平衡效果与参数量）
dropout = 0.2                          # Dropout 概率（防止小数据集过拟合）

# ===================== 优化器配置 =====================
learning_rate = 1e-3                   # 初始学习率（小模型可适当提高）
max_iters = 5000                       # 总训练步数（58000首诗适配）
lr_decay_iters = 5000                  # 学习率衰减步数（与总步数一致）
min_lr = 1e-4                          # 最小学习率（初始学习率/10）
beta2 = 0.99                           # AdamW 二阶矩衰减率（小批量优化）
warmup_iters = 100                     # 学习率预热步数（稳定初始训练）
```

<img width="434" height="196" alt="image" src="https://github.com/user-attachments/assets/3349ecff-98b9-4820-9cb3-427699e3a909" />

### 4.模型训练
```
python train.py config/train_poemtext_char.py
```

<img width="947" height="715" alt="屏幕截图 2026-03-19 132046" src="https://github.com/user-attachments/assets/45ab1f14-13c6-422a-8aa5-ffafee21d0e0" />
模型训练结束后，会在项⽬⽬录下⽣成⼀个⽂件权重out-poemtext-chat/ckpt.pt，该模型权重就是由58000⾸诗词tang_poet.txt数据得到的GPT

### 5.模型的推理及采样
```
python sample.py --out_dir=out-poemtext-char
```

<img width="1460" height="507" alt="屏幕截图 2026-03-19 134345" src="https://github.com/user-attachments/assets/8fd39e35-a1c5-466e-b2be-1aeec36f70de" />

| 参数           | 作用                     | 推荐值  |
| -------------- | ------------------------ | ------- |
| start          | 诗词开头文本             | 自定义  |
| max_new_tokens | 生成最大字符数           | 256     |
| temperature    | 生成随机性（越小越严谨） | 0.7-0.9 |
| top_k          | 采样候选词数量           | 200     |

### 6.常见问题

| 问题现象       | 原因分析                | 解决方案                                     |
| -------------- | ----------------------- | -------------------------------------------- |
| 训练时显存不足 | batch_size 过大         | 将 batch_size 改为 32/16，或启用梯度累积     |
| 生成诗词不通顺 | temperature 过高 / 过低 | 调整为 0.7-0.9，或增大 top_k 至 200-300      |
| 预处理脚本报错 | 语料编码非 UTF-8        | 将语料转为 UTF-8 编码，删除特殊字符          |
| 训练损失不下降 | 学习率不合适 / 步数不足 | 调整 learning_rate 为 5e-4，或增加 max_iters |

### <进阶优化建议

1. 提升生成质量

   增加训练步数（max_iters=10000），但需监控过拟合

   微调 temperature=0.75 + top_k=250，平衡创意与通顺

2. 加速训练

   有 NVIDIA GPU 时，安装 CUDA 版本 PyTorch，训练速度提升 5-10 倍

   启用梯度累积（gradient_accumulation_steps=2），模拟更大批次

3. 自定义语料

   可替换为宋词 / 元曲语料，仅需修改 `tang_poet.txt` 并重新预处理

### 思考题

1. **使⽤《天⻰⼋部》tianlong.txt数据集训练⼀个GPT模型，并⽣成内容看看效果。**

   要基于《天龙八部》数据集训练一个字符级 GPT 模型，我们可以复用 nanoGPT 的完整流程。首先需要将 `tianlong.txt` 放入 `data/tianlong` 目录，仿照 `poemtext` 数据集的 `prepare.py` 编写预处理脚本：读取文本、构建字符词汇表、将全文编码为整数序列，再按 9:1 比例切分为训练集和验证集，保存为 `train.bin` 和 `val.bin`。接下来复制 `config/train_poemtext_char.py` 为新的训练配置，将数据集路径修改为 `data/tianlong`，并根据硬件条件调整 `batch_size`、`block_size` 等参数 —— 比如显存不足时可以适当减小批次大小和上下文窗口长度。启动训练后，模型会通过自回归目标学习武侠文本的语言模式，逐字符预测下一个字的概率分布。训练完成后，修改 `sample.py` 中的模型路径和起始提示（如 “乔峰喝道：”），执行采样脚本即可生成续写文本。我们可以观察生成内容是否贴合原著文风：比如人物对话的语气、武功招式的描述、情节推进的逻辑，以此评估模型对《天龙八部》世界观和语言特征的学习效果。

2. **思考config/train_poemtext_char.py训练⽂件中其他参数的含义，修改其参数并重新进⾏训练，看看其效果怎么**
   **样。**

  `config/train_poemtext_char.py` 中的参数决定了模型结构、训练策略和资源消耗，核心参数及其调参效果如下：

  - `n_layer`：Transformer 编码器的层数，决定模型的深度。增大该值可以提升模型捕捉复杂语义的能力，但会显著增加参数量和计算量，容易导致过拟合；减小则模型更轻量化，训练速度更快，但对长距离依赖的建模能力会减弱。

    `n_head`：多头注意力的头数，必须能整除 `n_embd`。增加头数可以让模型从多个维度捕捉文本特征，比如同时关注字词搭配和句式节奏，但也会提升显存占用和计算开销。

    `n_embd`：词向量维度，控制每个字符的语义表达精度。维度越高，模型能编码的细节越丰富，生成文本的连贯性越强，但参数量会呈平方级增长，对显存要求更高。

    `block_size`：上下文窗口长度，即模型一次能看到的最大字符数。增大该值可以让模型学习更长的文本依赖（比如诗句的对仗、段落的逻辑连贯），但会显著增加显存占用和训练时间。

    `learning_rate`：控制参数更新的步长。学习率过大容易导致训练震荡、不收敛；过小则训练进度缓慢，模型容易陷入局部最优。通常需要配合学习率衰减策略，在训练后期逐步降低学习率以稳定收敛。

  - `dropout`：Dropout 比例，用于防止过拟合。值越大，正则化效果越强，训练过程更稳定，但会降低模型的拟合能力；值过小则容易在小数据集上出现过拟合，生成文本重复、刻板。

  通过调整这些参数，我们可以观察模型性能的变化：比如减小 `n_layer` 和 `n_embd` 可以在资源有限的情况下快速训练，但生成文本的复杂度会下降；增大 `block_size` 能提升长文本生成的连贯性，但需要更高的硬件配置。

3. **研究模型采样⽂件sample.py，思考模型的推理过程是怎么样的？**

`sample.py` 是模型的推理采样脚本，完整实现了自回归生成流程：

1. 加载与初始化：首先从训练输出目录加载模型权重和字符级词汇表（`meta.pkl`），将模型切换到评估模式（`model.eval()`），关闭 Dropout 等训练特有的正则化操作。同时将用户输入的起始提示（如 “段誉道：”）编码为整数 token 序列，作为模型的初始输入。
2. 逐 token 生成：进入循环后，模型每次接收当前输入序列的最后 `block_size` 个 token，通过前向传播得到下一个 token 的概率分布。为了平衡生成质量和多样性，代码会通过 `temperature` 参数缩放概率分布：温度越低，概率分布越尖锐，生成文本越保守、贴近训练数据；温度越高，分布越平缓，生成内容越随机、富有创意。同时通过 `top_k` 采样限制候选范围，只保留概率最高的 k 个 token，避免生成无意义的低频字符。
3. 序列拼接与解码：从筛选后的概率分布中采样一个 token，追加到输入序列末尾，重复该过程直到生成指定长度的文本。最后将生成的整数 token 序列通过词汇表解码为可读的中文文本，输出最终结果。

整个推理过程的核心是自回归生成：模型基于已生成的文本片段，逐字预测下一个最可能的字符，完全依赖训练阶段学到的语言模式和上下文依赖关系，最终实现续写、创作等文本生成任务。
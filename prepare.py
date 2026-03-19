import os
import requests  # 你代码中导入但未使用，若不需要可删除
import tiktoken
import numpy as np

# 1. 定义文件路径
input_file_path = 'tang_poet.txt'

# 2. 检查文件是否存在（避免FileNotFoundError）
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"错误：未找到文件 {input_file_path}，请检查文件路径是否正确")

# 3. 关键修复：指定UTF-8编码读取，添加容错处理
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()

# 4. 过滤无效空白字符（可选，避免编码残留问题）
data = data.replace('\u3000', '').replace('\xa0', '').strip()

# 5. 划分训练/验证集（你的原有逻辑，无问题）
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 6. GPT2编码（你的原有逻辑，无问题）
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

# 7. 打印token数量（你的原有逻辑，格式优化）
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 8. 保存为二进制文件（你的原有逻辑，无问题）
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__),'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__),'val.bin'))

print("✅ 数据预处理完成！已生成 train.bin 和 val.bin")
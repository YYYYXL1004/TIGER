# -*- coding: utf-8 -*-
"""
本脚本由 process.ipynb 转换而来，实现数据预处理、embedding生成等功能。
"""

# ========================
# Convert raw data to 'strict' json
# ========================
import json
import gzip
import os
import numpy as np
import pandas as pd

# 数据集名称
# 如需处理其他数据集请修改此处
# ------------------------
dataset_name = "Beauty"
os.makedirs(dataset_name, exist_ok=True)

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

# 处理原始gz数据，转为严格json
with open(f"./{dataset_name}/{dataset_name}.json", 'w') as f:
    for l in parse(f"reviews_{dataset_name}_5.json.gz"):
        f.write(l + '\n')

# 打印行数和首行内容
with open(f"./{dataset_name}/{dataset_name}.json", 'r') as data:
    print("Number of lines:", sum(1 for _ in data))
    data.seek(0)
    print("First line:", data.readline().strip())

# ========================
# 用户/物品ID映射与数据分割
# ========================
userID_mapping = {}
itemID_mapping = {}

with open(f"./{dataset_name}/{dataset_name}.json", 'r') as data:
    userIDs = []
    itemIDs = []
    timestamps = []
    for line in data:
        review = json.loads(line.strip())
        userID = review['reviewerID']
        itemID = review['asin']
        timestamp = review['unixReviewTime']
        if userID not in userID_mapping:
            userID_mapping[userID] = len(userID_mapping) + 1
        if itemID not in itemID_mapping:
            itemID_mapping[itemID] = len(itemID_mapping) + 1
        userIDs.append(userID_mapping[userID])
        itemIDs.append(itemID_mapping[itemID])
        timestamps.append(timestamp)

np.save(f'./{dataset_name}/user_mapping.npy', userID_mapping)
print("user_num:", len(userID_mapping))
print("the first five userID mapping:", list(userID_mapping.items())[:5])
np.save(f'./{dataset_name}/item_mapping.npy', itemID_mapping)
print("item_num:", len(itemID_mapping))
print("the first five itemID mapping:", list(itemID_mapping.items())[:5])

# 用户-物品序列按时间排序
def group_user_items(userIDs, itemIDs, timestamps):
    user_item_mapping = {}
    for userID, itemID, timestamp in zip(userIDs, itemIDs, timestamps):
        if userID not in user_item_mapping:
            user_item_mapping[userID] = []
        user_item_mapping[userID].append((itemID, timestamp))
    for userID in user_item_mapping:
        user_item_mapping[userID].sort(key=lambda x: x[1])
        user_item_mapping[userID] = [item[0] for item in user_item_mapping[userID]]
    return user_item_mapping

user_item_mapping = group_user_items(userIDs, itemIDs, timestamps)
print("user-item mapping:", list(user_item_mapping.items())[:5])

# leave-one-out分割
def split_data(user_item_mapping):
    train_data = {}
    val_data = {}
    test_data = {}
    for userID, item_sequence in user_item_mapping.items():
        train_data[userID] = item_sequence[:-2]
        val_data[userID] = item_sequence[:-1]
        test_data[userID] = item_sequence
    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(user_item_mapping)
print("training data:", list(train_data.items())[:5])
print("validation data:", list(val_data.items())[:5])
print("testing data:", list(test_data.items())[:5])

def prepare_data(data_dict):
    rows = []
    for userID, item_sequence in data_dict.items():
        history = item_sequence[:-1]
        target = item_sequence[-1]
        rows.append({'user': userID, 'history': history, 'target': target})
    return pd.DataFrame(rows)

train_df = prepare_data(train_data)
print("\nTraining data shape:", train_df.shape)
print("the first 3 rows of training data:\n", train_df.head(3))
val_df = prepare_data(val_data)
print("\nValidation data shape:", val_df.shape)
print("the first 3 rows of validation data:\n", val_df.head(3))
test_df = prepare_data(test_data)
print("\nTesting data shape:", test_df.shape)
print("the first 3 rows of testing data:\n", test_df.head(3))

train_df.to_parquet(f'./{dataset_name}/train.parquet', index=False)
val_df.to_parquet(f'./{dataset_name}/valid.parquet', index=False)
test_df.to_parquet(f'./{dataset_name}/test.parquet', index=False)
print("Data saved to parquet files.")

# ========================
# 生成Item元数据文件
# ========================
with open(f"./{dataset_name}/{dataset_name}_metadata.json", 'w') as f:
    for l in parse(f"meta_{dataset_name}.json.gz"):
        f.write(l + '\n')

# 打印前5个item元数据
with open(f"./{dataset_name}/{dataset_name}_metadata.json", 'r') as metadata_file:
    reverse_itemID_mapping = {v: k for k, v in itemID_mapping.items()}
    item_info = {}
    for line in metadata_file:
        metadata = json.loads(line.strip())
        asin = metadata.get('asin')
        if asin in reverse_itemID_mapping.values():
            itemID = itemID_mapping[asin]
            item_info[itemID] = {
                'title': metadata.get('title') if metadata.get('title') else None,
                'price': metadata.get('price') if metadata.get('price') else None,
                'salesRank': metadata.get('salesRank') if metadata.get('salesRank') else None,
                'brand': metadata.get('brand') if metadata.get('brand') else None,
                'categories': metadata.get('categories') if metadata.get('categories') else None,
            }
for itemID, info in list(item_info.items())[:5]:
    print(f"ItemID: {itemID}, Info: {info}")

# ========================
# 生成Item语义Embedding
# ========================
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 如需下载模型请参考注释
# modelscope download --model sentence-transformers/sentence-t5-base  --local_dir ./dir
model = SentenceTransformer('../sentence-t5-base')

item_embeddings = []
print("Generating embeddings for ALL items found in reviews...")
for asin, itemID in tqdm(itemID_mapping.items()):
    # 尝试从 item_info 获取元数据，如果获取不到（说明meta文件里没有），就用一个空字典
    info = item_info.get(itemID, {})
    
    # 用获取到的info（可能为空）来构建描述字符串
    semantics = f"'title':{info.get('title', '')}\n 'price':{info.get('price', '')}\n 'salesRank':{info.get('salesRank', '')}\n 'brand':{info.get('brand', '')}\n 'categories':{info.get('categories', '')}"
    embedding = model.encode(semantics)
    item_embeddings.append({'ItemID': itemID, 'embedding': embedding.tolist()})
# for itemID, info in item_info.items():
#     semantics = f"'title':{info.get('title', '')}\n 'price':{info.get('price', '')}\n 'salesRank':{info.get('salesRank', '')}\n 'brand':{info.get('brand', '')}\n 'categories':{info.get('categories', '')}"
#     embedding = model.encode(semantics)
#     item_embeddings.append({'ItemID': itemID, 'embedding': embedding.tolist()})

item_emb_df = pd.DataFrame(item_embeddings)
print("\nItem embeddings DataFrame shape:", item_emb_df.shape)
print("The first 3 rows of item embeddings DataFrame:\n", item_emb_df.head(3))
item_emb_df.to_parquet(f'./{dataset_name}/item_emb.parquet', index=False)
print("Item embeddings saved to item_emb.parquet.")
# embeddings = np.array([item['embedding'] for item in item_embeddings])
# np.save(f'./{dataset_name}/item_emb.npy', embeddings)
# print("Item embeddings saved to item_emb.npy.") 
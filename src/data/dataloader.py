import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

#定义数据加载器
class GPTDatasetV1(Dataset):
    #初始化
    def __init__(self,txt,tokenizer,max_length,stride):
        #创建输入输出对
        self.input_ids = []
        self.target_ids = []

        #encode.....tokenizer实例将在之后被创建
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        #滑动窗口并生成输入输出tensor对
        for i in range(0,len(token_ids) - max_length,stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader_v1(
     txt,            # 原始文本（比如 the-verdict.txt）
     batch_size=4,   # 一次喂 4 组数据给模型
     max_length=256, # 每段数据最长 256 个词元
     stride=128,     # 每次滑动 128 个词元（窗口重叠）
     shuffle=True,   # 训练前打乱顺序
     drop_last=True, # 最后不够1个batch就丢掉
     num_workers=0   # 加载数据用的线程
    ):
     tokenizer = tiktoken.get_encoding("gpt2")

     #数据集
     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

     #数据加载器
     dataloader = DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle,
         drop_last=drop_last,
         num_workers=num_workers
     )

     return dataloader

#1.读取文件
with open("data/the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

#2.初始化嵌入层
vocab_size = 50257#词元id数
output_dim = 256#嵌入维度
context_length = 1024#最长输入文本

#词元嵌入
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
#位置嵌入
pos_embedding_layer = torch.nn.Embedding(context_length,output_dim)

#3.训练嵌入层，将词元从ID转化为包含（词元+位置）信息的可用嵌入向量
batch_size = 8
max_length = 4
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=batch_size,
    max_length=max_length,
    stride=max_length
)

for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings

    break

print (input_embeddings.shape)

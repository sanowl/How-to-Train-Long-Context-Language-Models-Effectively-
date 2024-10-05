import os, torch, math, openai, numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from flash_attn_2 import flash_attn_func
from mosaic_streaming import StreamingDataset
from deepspeed.runtime.pipe import PipelineModule
import deepspeed
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import ndcg_score

MAX_SEQ_LEN_STAGE1, MAX_SEQ_LEN_STAGE2 = 65536, 524288
TOTAL_TOKENS_STAGE1, TOTAL_TOKENS_STAGE2 = 20*1024**3, 20*1024**3
BATCH_SIZE_STAGE1, BATCH_SIZE_STAGE2 = 4*1024**2, 8*1024**2
ROPE_BASE_STAGE1, ROPE_BASE_STAGE2 = 8*10**6, 128*10**6

class ProLongDataset(StreamingDataset):
    def __init__(self, code_repos, books, textbooks, shortmix, max_len):
        super().__init__()
        self.code_repos, self.books, self.textbooks, self.shortmix = code_repos, books, textbooks, shortmix
        self.max_len = max_len
        self.shortmix_types = ['FineWeb']*27 + ['FineWeb-Edu']*27 + ['Wikipedia']*8 + ['Tulu-v2']*11 + ['StackExchange']*11 + ['ArXiv']*8 + ['OpenWebMath']*8
    def __len__(self):
        return len(self.code_repos) + len(self.books) + len(self.textbooks) + len(self.shortmix)
    def __getitem__(self, idx):
        if idx < len(self.code_repos): return self.process_long_data(self.code_repos[idx], 'code_repos')
        elif idx < len(self.code_repos)+len(self.books): return self.process_long_data(self.books[idx-len(self.code_repos)], 'books')
        elif idx < len(self.code_repos)+len(self.books)+len(self.textbooks): return self.process_long_data(self.textbooks[idx-len(self.code_repos)-len(self.books)], 'textbooks')
        else:
            sm_type = np.random.choice(self.shortmix_types)
            return self.shortmix[np.random.randint(len(self.shortmix))], sm_type
    def process_long_data(self, data, dtype):
        if self.max_len == MAX_SEQ_LEN_STAGE2:
            use_full = (dtype=='code_repos' and torch.rand(1).item()<0.5) or (dtype=='books' and torch.rand(1).item()<0.17) or (dtype=='textbooks')
            return data[:self.max_len] if use_full else data[:MAX_SEQ_LEN_STAGE1], dtype
        return data[:self.max_len], dtype

def smart_collate(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    texts, types = zip(*batch)
    return texts, types

def create_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=smart_collate)

def rope(x, dim, base):
    device, d = x.device, x.shape[-1]
    pos = torch.arange(dim, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device)*(-math.log(base)/d))
    sc = torch.stack([torch.sin(pos*div), torch.cos(pos*div)], dim=-1)
    emb = torch.zeros_like(x)
    emb[...,::2], emb[...,1::2] = sc[...,0], sc[...,1]
    return x*emb.cos() + torch.roll(x, shifts=1, dims=-1)*emb.sin()

def create_document_mask(lengths, max_len):
    mask = torch.zeros(len(lengths), max_len, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths): mask[i, :l, :l] = True
    return mask

class ProLongModel(PipelineModule):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits

def train_stage(model, dataloader, optimizer, scheduler, device, total_tokens, max_seq_len, rope_base, token_avg=False):
    model.train()
    tokens, total_loss = 0, 0
    for batch, dtype in dataloader:
        if tokens >= total_tokens: break
        optimizer.zero_grad()
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len).to(device)
        inputs['input_ids'] = rope(inputs['input_ids'], max_seq_len, rope_base)
        mask = create_document_mask(inputs['attention_mask'].sum(1).tolist(), inputs['input_ids'].shape[1]).to(device)
        outputs = model(input_ids=inputs['input_ids'], attention_mask=mask).logits
        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), inputs['input_ids'].view(-1), reduction='mean' if token_avg else 'sum')
        loss.backward()
        optimizer.step()
        scheduler.step()
        tokens += inputs['input_ids'].numel()
        total_loss += loss.item()
    return total_loss / tokens if token_avg else total_loss

def load_ultrachat_data():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    return ds.map(lambda ex: {"text": " ".join([m["content"] for m in ex["messages"]])}, remove_columns=ds["train"].column_names)

def evaluate_model(model, tokenizer, task, data):
    model.eval()
    scores = []
    for item in tqdm(data):
        context = item.get('context') or item.get('question') or ''
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = tokenizer.decode(model.generate(**inputs, max_new_tokens=100)[0], skip_special_tokens=True)
        if task in ['qa', 'summarization']:
            score = gpt4_evaluate(task, item, generated)
        elif task in ['recall', 'rag', 'icl']:
            score = evaluate_task(task, item, generated)
        elif task == 're-rank':
            score = ndcg_score([item['relevance']], [item['score']])
        scores.append(score)
    return np.mean(scores)

def gpt4_evaluate(task, item, gen):
    if task == 'qa':
        q, ref = item['question'], item['answer']
        prompt = f"Question: {q}\nReference: {ref}\nGenerated: {gen}\nScore (0-1):"
    else:
        ref = item['reference']
        prompt = f"Reference: {ref}\nGenerated: {gen}\nScore (0-1):"
    response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role":"system","content":"Evaluate the response."},{"role":"user","content":prompt}], max_tokens=1)
    return float(response['choices'][0]['message']['content'].strip())

def evaluate_task(task, item, gen):
    if task in ['recall', 'rag']:
        return 1.0 if gen.strip() == item.get('answer', item.get('target', '')).strip() else 0.0
    if task == 'icl':
        return 1.0 if gen.strip() == item.get('label', '').strip() else 0.0
    return 0.0

def load_task_data(task):
    datasets_map = {
        'rag': ["natural_questions", "hotpot_qa", "popqa"],
        're-rank': ["msmarco"],
        'icl': ["trec_coarse", "trec_fine", "nlu", "banking77", "clinc150"],
        'qa': ["narrativeqa"],
        'summarization': ["multi_lexsum"]
    }
    if task in datasets_map:
        return [load_dataset(ds)['test'] if 'test' in load_dataset(ds) else load_dataset(ds)['validation'] for ds in datasets_map[task]]
    return []

code_repos, books, textbooks, shortmix = ["code1", "code2"], ["book1", "book2"], ["textbook1"], ["short1", "short2"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b-instruct")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b-instruct")
for layer in base_model.model.layers: layer.self_attn.forward = flash_attn_func
model = ProLongModel(base_model)
ds_config = {"train_batch_size": BATCH_SIZE_STAGE1, "fp16": {"enabled": True}, "zero_optimization": {"stage":3,"overlap_comm":True,"contiguous_gradients":True}, "pipeline": {"stages": "auto", "partition": "best"}, "sequence_parallel": {"enabled": True}}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9,0.95), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_TOKENS_STAGE1//BATCH_SIZE_STAGE1, eta_min=1e-6)
model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)
dataset_64k = ProLongDataset(code_repos, books, textbooks, shortmix, MAX_SEQ_LEN_STAGE1)
dataloader_64k = create_dataloader(dataset_64k, BATCH_SIZE_STAGE1//model.world_size)
train_stage(model, dataloader_64k, optimizer, scheduler, model.device, TOTAL_TOKENS_STAGE1, MAX_SEQ_LEN_STAGE1, ROPE_BASE_STAGE1)
model.base_model.config.max_position_embeddings = MAX_SEQ_LEN_STAGE2
dataset_512k = ProLongDataset(code_repos, books, textbooks, shortmix, MAX_SEQ_LEN_STAGE2)
dataloader_512k = create_dataloader(dataset_512k, BATCH_SIZE_STAGE2//model.world_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9,0.95), weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_TOKENS_STAGE2//BATCH_SIZE_STAGE2, eta_min=1e-6)
train_stage(model, dataloader_512k, optimizer, scheduler, model.device, TOTAL_TOKENS_STAGE2, MAX_SEQ_LEN_STAGE2, ROPE_BASE_STAGE2)
sft_data = load_ultrachat_data()
sft_dataloader = create_dataloader(sft_data, 4*1024**2//model.world_size)
sft_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9,0.95), weight_decay=0.1)
sft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sft_optimizer, T_max=1_000_000_000//(4*1024**2), eta_min=2e-6)
train_stage(model, sft_dataloader, sft_optimizer, sft_scheduler, model.device, 1_000_000_000, MAX_SEQ_LEN_STAGE2, ROPE_BASE_STAGE2, token_avg=True)
evaluation_tasks = ['recall','rag','re-rank','icl','qa','summarization']
for task in evaluation_tasks:
    data = load_task_data(task)
    for ds in data:
        score = evaluate_model(model, tokenizer, task, ds)
        print(f"{task} score: {score}")

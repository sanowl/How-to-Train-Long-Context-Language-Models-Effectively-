import os
import torch
import math
import openai
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    Auto
    dModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from flash_attn_2 import flash_attn_func
from mosaic_streaming import StreamingDataset
from deepspeed.runtime.pipe import PipelineModule
import deepspeed
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from sklearn.metrics import ndcg_score
import logging
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    max_seq_len_stage1: int = 65536
    max_seq_len_stage2: int = 524288
    total_tokens_stage1: int = 20 * 1024**3
    total_tokens_stage2: int = 20 * 1024**3
    batch_size_stage1: int = 4 * 1024**2
    batch_size_stage2: int = 8 * 1024**2
    batch_size_sft: int = 4 * 1024**2
    rope_base_stage1: int = 8 * 10**6
    rope_base_stage2: int = 128 * 10**6
    learning_rate_stage1: float = 1e-5
    learning_rate_stage2: float = 1e-5
    learning_rate_sft: float = 2e-5
    warmup_ratio: float = 0.1
    eta_min: float = 1e-6
    eta_min_sft: float = 2e-6
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints/"
    save_every: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "meta-llama/Llama-3-8b-instruct"

config = TrainingConfig()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

class ProLongDataset(StreamingDataset):
    def __init__(self, code_repos, books, textbooks, shortmix, max_len):
        super().__init__()
        self.code_repos = code_repos
        self.books = books
        self.textbooks = textbooks
        self.shortmix = shortmix
        self.max_len = max_len
        type_counts = {
            'FineWeb': 27,
            'FineWeb-Edu': 27,
            'Wikipedia': 8,
            'Tulu-v2': 11,
            'StackExchange': 11,
            'ArXiv': 8,
            'OpenWebMath': 8
        }
        self.shortmix_types = [stype for stype, count in type_counts.items() for _ in range(count)]

    def __len__(self):
        return len(self.code_repos) + len(self.books) + len(self.textbooks) + len(self.shortmix)

    def __getitem__(self, idx):
        if idx < len(self.code_repos):
            return self.process_long_data(self.code_repos[idx], 'code_repos')
        elif idx < len(self.code_repos) + len(self.books):
            return self.process_long_data(self.books[idx - len(self.code_repos)], 'books')
        elif idx < len(self.code_repos) + len(self.books) + len(self.textbooks):
            return self.process_long_data(self.textbooks[idx - len(self.code_repos) - len(self.books)], 'textbooks')
        else:
            sm_type = np.random.choice(self.shortmix_types)
            return self.shortmix[np.random.randint(len(self.shortmix))], sm_type

    def process_long_data(self, data, dtype):
        if self.max_len == config.max_seq_len_stage2:
            use_full = (
                (dtype == 'code_repos' and torch.rand(1).item() < 0.5) or
                (dtype == 'books' and torch.rand(1).item() < 0.17) or
                (dtype == 'textbooks')
            )
            return data[:self.max_len] if use_full else data[:config.max_seq_len_stage1], dtype
        return data[:self.max_len], dtype

def smart_collate(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    texts, types = zip(*batch)
    return texts, types

def create_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=smart_collate,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

class ProLongModel(PipelineModule):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, base_model
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise

def rope(x, dim, base):
    pass

def create_document_mask(lengths, max_len):
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return mask

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}.pt'))
    logging.info(f"Checkpoint saved at epoch {epoch}")

def load_ultrachat_data():
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k")
        return ds.map(
            lambda ex: {"text": " ".join([m["content"] for m in ex["messages"]])},
            remove_columns=ds["train"].column_names
        )
    except Exception as e:
        logging.error(f"Error loading ultrachat data: {e}")
        return None

def gpt4_evaluate(task, item, gen):
    if task == 'qa':
        q, ref = item.get('question', ''), item.get('answer', '')
        prompt = f"Question: {q}\nReference: {ref}\nGenerated: {gen}\nScore (0-1):"
    else:
        ref = item.get('reference', '')
        prompt = f"Reference: {ref}\nGenerated: {gen}\nScore (0-1):"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Evaluate the response."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1
        )
        score = float(response['choices'][0]['message']['content'].strip())
        return score
    except Exception as e:
        logging.error(f"GPT-4 evaluation error: {e}")
        return 0.0

def evaluate_task(task, item, gen):
    if task in ['recall', 'rag']:
        return 1.0 if gen.strip() == item.get('answer', item.get('target', '')).strip() else 0.0
    if task == 'icl':
        return 1.0 if gen.strip() == item.get('label', '').strip() else 0.0
    return 0.0

def evaluate_model(model, tokenizer, task, data, batch_size=32):
    model.eval()
    scores = []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {task}"):
        batch = data[i:i+batch_size]
        contexts = [item.get('context') or item.get('question') or '' for item in batch]
        try:
            inputs = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True).to(config.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
            generated_texts = [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated_ids]
            for j, gen in enumerate(generated_texts):
                item = batch[j]
                if task in ['qa', 'summarization']:
                    score = gpt4_evaluate(task, item, gen)
                elif task in ['recall', 'rag', 'icl']:
                    score = evaluate_task(task, item, gen)
                elif task == 're-rank':
                    score = ndcg_score([item.get('relevance', [0])], [item.get('score', [0])])
                else:
                    score = 0.0
                scores.append(score)
        except Exception as e:
            logging.error(f"Evaluation error at batch {i}: {e}")
            continue
    mean_score = np.mean(scores) if scores else 0.0
    logging.info(f"{task} score: {mean_score}")
    return mean_score

def load_task_data(task):
    datasets_map = {
        'rag': ["natural_questions", "hotpot_qa", "popqa"],
        're-rank': ["msmarco"],
        'icl': ["trec_coarse", "trec_fine", "nlu", "banking77", "clinc150"],
        'qa': ["narrativeqa"],
        'summarization': ["multi_lexsum"]
    }
    loaded_datasets = []
    if task in datasets_map:
        for ds_name in datasets_map[task]:
            try:
                ds = load_dataset(ds_name)
                if 'test' in ds:
                    loaded_datasets.append(ds['test'])
                elif 'validation' in ds:
                    loaded_datasets.append(ds['validation'])
                else:
                    loaded_datasets.append(ds['train'])
            except Exception as e:
                logging.error(f"Error loading dataset {ds_name} for task {task}: {e}")
    return loaded_datasets

def train_stage(model, dataloader, optimizer, scheduler, device, total_tokens, max_seq_len, rope_base, token_avg=False, stage_name=""):
    model.train()
    tokens, total_loss = 0, 0
    for batch_idx, (batch, dtype) in enumerate(dataloader):
        if tokens >= total_tokens:
            logging.info(f"Reached total tokens limit for {stage_name} stage.")
            break
        optimizer.zero_grad()
        try:
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len
            ).to(device)
        except Exception as e:
            logging.error(f"Tokenization error at batch {batch_idx}: {e}")
            continue

        attention_mask = inputs['attention_mask']
        try:
            outputs = model(input_ids=inputs['input_ids'], attention_mask=attention_mask)
            logits = outputs
        except Exception as e:
            logging.error(f"Model forward pass error at batch {batch_idx}: {e}")
            continue

        try:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                inputs['input_ids'].view(-1),
                reduction='mean' if token_avg else 'sum'
            )
        except Exception as e:
            logging.error(f"Loss computation error at batch {batch_idx}: {e}")
            continue

        try:
            loss.backward()
            optimizer.step()
            scheduler.step()
        except Exception as e:
            logging.error(f"Optimizer/Scheduler step error at batch {batch_idx}: {e}")
            continue

        tokens += inputs['input_ids'].numel()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            logging.info(f"[{stage_name} Stage] Batch {batch_idx}: Loss = {loss.item()}")

        if batch_idx % config.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, batch_idx, loss.item(), config.checkpoint_dir)

    average_loss = total_loss / tokens if token_avg else total_loss
    logging.info(f"[{stage_name} Stage] Average Loss: {average_loss}")
    return average_loss

def main():
    try:
        tokenizer, base_model = load_model(config.model_name)

        try:
            for layer in base_model.model.layers:
                layer.self_attn.forward = flash_attn_func
            logging.info("Successfully replaced self-attention with FlashAttention.")
        except AttributeError as e:
            logging.error(f"Error replacing self-attention layers: {e}")
            return

        model = ProLongModel(base_model)
        model.to(config.device)

        ds_config = {
            "train_batch_size": config.batch_size_stage1,
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "pipeline": {
                "stages": "auto",
                "partition": "best"
            },
            "sequence_parallel": {"enabled": True}
        }

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate_stage1,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(config.warmup_ratio * (config.total_tokens_stage1 // config.batch_size_stage1)),
            num_training_steps=(config.total_tokens_stage1 // config.batch_size_stage1)
        )

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config_params=ds_config
        )

        code_repos = ["code1", "code2"]
        books = ["book1", "book2"]
        textbooks = ["textbook1"]
        shortmix = ["short1", "short2"]

        dataset_stage1 = ProLongDataset(
            code_repos=code_repos,
            books=books,
            textbooks=textbooks,
            shortmix=shortmix,
            max_len=config.max_seq_len_stage1
        )
        dataloader_stage1 = create_dataloader(
            dataset_stage1,
            batch_size=config.batch_size_stage1 // model_engine.world_size
        )

        logging.info("Starting Stage 1 Training")
        train_stage(
            model=model_engine,
            dataloader=dataloader_stage1,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.device,
            total_tokens=config.total_tokens_stage1,
            max_seq_len=config.max_seq_len_stage1,
            rope_base=config.rope_base_stage1,
            stage_name="Stage 1"
        )

        try:
            model_engine.base_model.config.max_position_embeddings = config.max_seq_len_stage2
            logging.info("Updated model's max_position_embeddings for Stage 2.")
        except AttributeError as e:
            logging.error(f"Error updating max_position_embeddings: {e}")

        dataset_stage2 = ProLongDataset(
            code_repos=code_repos,
            books=books,
            textbooks=textbooks,
            shortmix=shortmix,
            max_len=config.max_seq_len_stage2
        )
        dataloader_stage2 = create_dataloader(
            dataset_stage2,
            batch_size=config.batch_size_stage2 // model_engine.world_size
        )

        optimizer_stage2 = torch.optim.AdamW(
            model_engine.parameters(),
            lr=config.learning_rate_stage2,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        scheduler_stage2 = get_cosine_schedule_with_warmup(
            optimizer_stage2,
            num_warmup_steps=int(config.warmup_ratio * (config.total_tokens_stage2 // config.batch_size_stage2)),
            num_training_steps=(config.total_tokens_stage2 // config.batch_size_stage2)
        )

        model_engine.optimizer = optimizer_stage2
        model_engine.lr_scheduler = scheduler_stage2

        logging.info("Starting Stage 2 Training")
        train_stage(
            model=model_engine,
            dataloader=dataloader_stage2,
            optimizer=optimizer_stage2,
            scheduler=scheduler_stage2,
            device=config.device,
            total_tokens=config.total_tokens_stage2,
            max_seq_len=config.max_seq_len_stage2,
            rope_base=config.rope_base_stage2,
            stage_name="Stage 2"
        )

        sft_data = load_ultrachat_data()
        if sft_data:
            sft_dataloader = create_dataloader(
                sft_data['train'],
                batch_size=config.batch_size_sft // model_engine.world_size
            )
            optimizer_sft = torch.optim.AdamW(
                model_engine.parameters(),
                lr=config.learning_rate_sft,
                betas=(0.9, 0.95),
                weight_decay=0.1
            )
            scheduler_sft = get_cosine_schedule_with_warmup(
                optimizer_sft,
                num_warmup_steps=int(config.warmup_ratio * (1_000_000_000 // config.batch_size_sft)),
                num_training_steps=(1_000_000_000 // config.batch_size_sft)
            )
            model_engine.optimizer = optimizer_sft
            model_engine.lr_scheduler = scheduler_sft

            logging.info("Starting SFT Training")
            train_stage(
                model=model_engine,
                dataloader=sft_dataloader,
                optimizer=optimizer_sft,
                scheduler=scheduler_sft,
                device=config.device,
                total_tokens=1_000_000_000,
                max_seq_len=config.max_seq_len_stage2,
                rope_base=config.rope_base_stage2,
                token_avg=True,
                stage_name="SFT"
            )
        else:
            logging.warning("SFT data not loaded. Skipping SFT training.")

        evaluation_tasks = ['recall', 'rag', 're-rank', 'icl', 'qa', 'summarization']
        for task in evaluation_tasks:
            data_splits = load_task_data(task)
            for ds in data_splits:
                score = evaluate_model(model_engine, tokenizer, task, ds)
                logging.info(f"{task} score: {score}")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()

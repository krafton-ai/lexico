import torch
import numpy as np
import random
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import Buffer
from model import Autoencoder
import argparse
import pprint
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Train layer-specific KV dictionaries through direct gradient-based optimization")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or identifier of the pretrained model")
    parser.add_argument("--dictionary_size", type=int, default=4096, help="Size of the dictionary")
    parser.add_argument("--sparsity", type=int, default=8, help="Sparsity level for approximation")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval in epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lm_batch_size", type=int, default=16, help="Batch size for forward pass of language model")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--buffer_mult", type=int, default=384, help="Multiplier determining buffer size for KV storage")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--update_mode", type=str, choices=["gradient", "closed_form"], default="gradient", help="Method to update dictionary")
    return vars(parser.parse_args())

def main(cfg):
    SEED = cfg["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name_or_path'])
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name_or_path'], torch_dtype=torch.float16, device_map=cfg["device"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    cfg['num_hidden_layers'] = model.config.num_hidden_layers
    cfg['head_dim'] = model.config.head_dim
    cfg['num_key_value_heads'] = model.config.num_key_value_heads
    cfg["name"] = f'{cfg["model_name_or_path"].replace("/", "_")}_N_{cfg["dictionary_size"]}_s_{cfg["sparsity"]}'
    
    autoencoder = Autoencoder(cfg)

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}')

    texts_train = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["train"]["text"]
    texts_test = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["test"]["text"]
    buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer = Buffer(cfg, model, tokenizer, texts_test)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=0)

    batches_per_epoch = len(texts_train) // cfg["batch_size"]

    for epoch in range(cfg["num_epochs"]):
        for i in tqdm.trange(batches_per_epoch):
            kvs = buffer.next()

            if cfg["update_mode"] == "gradient":
                loss, k_hat, y = autoencoder(kvs)
                loss.backward()
                autoencoder.normalise_decoder_weights()
                optimizer.step()
                optimizer.zero_grad()
            else:
                # --- Closed-form approach ---
                # 1) Encode using OMP -> y
                with torch.no_grad():
                    y = autoencoder.encode(kvs)  # shape (B, L, n)
                # 2) Update self.D in closed-form
                autoencoder.closed_form_update(kvs, y)
                # 3) Compute the loss purely for logging
                loss, k_hat, _ = autoencoder(kvs)

            loss_value = loss.item()

            writer.add_scalar('Loss/train', loss.item(), epoch * batches_per_epoch + i)
            rel_recon_error = torch.mean((torch.norm(kvs - k_hat, dim=-1) / (torch.norm(kvs, dim=-1) + 1e-8))).item()
            writer.add_scalar('RelativeReconstructionError/train', rel_recon_error, epoch * batches_per_epoch + i)

            del loss, k_hat, y

        scheduler.step()
        autoencoder.save()
        print(f"Checkpoint saved at the end of epoch {epoch + 1}")
    
        if (epoch + 1) % cfg["eval_interval"] == 0:
            autoencoder.eval()
            with torch.no_grad():
                kvs = eval_buffer.next()
                loss, k_hat, y = autoencoder(kvs)
                rel_recon_error = torch.mean((torch.norm(kvs - k_hat, dim=-1) / (torch.norm(kvs, dim=-1) + 1e-8))).item()
                writer.add_scalar('Loss/eval', loss, epoch + 1)
                writer.add_scalar('RelativeReconstructionError/eval', rel_recon_error, epoch + 1)
            autoencoder.train()
    
    autoencoder.save_dictionary()
    print("Dictionary saved")

    writer.close()

if __name__ == "__main__":
    cfg = parse_args()
    pprint.pprint(cfg)
    main(cfg)
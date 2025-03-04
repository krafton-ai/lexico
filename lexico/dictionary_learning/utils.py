import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple

class BaseBuffer:
    """Base class for KV cache buffers with common functionality"""
    def __init__(self, cfg: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: List[str]):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.texts = dataset
        self.device = cfg["device"]
        self.buffer = torch.zeros((cfg["batch_size"] * cfg["buffer_mult"], cfg["num_hidden_layers"] * 2, cfg["head_dim"]), device=self.device)
        self.text_pointer = 0
        self.pointer = 0
        random.shuffle(self.texts)
        self.refresh()

    def next(self):
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out

    def _extract_kv_cache(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Extract and reshape KV cache from past_key_values"""
        kvs = []
        for l in range(self.cfg["num_hidden_layers"]):
            keys, values = past_key_values[l]
            kvs.append(keys)
            kvs.append(values)
        return torch.stack(kvs).permute(1, 3, 2, 0, 4).reshape(-1, self.cfg["num_hidden_layers"] * 2, self.cfg["head_dim"])

    def _create_attention_mask(self, input_ids: torch.Tensor, gen_sequences: torch.Tensor) -> torch.Tensor:
        """Create attention mask for both input and generated tokens"""
        input_mask = torch.ones_like(input_ids, dtype=torch.bool)
        gen_mask = torch.ones((input_ids.shape[0], gen_sequences.shape[1] - input_ids.shape[1]), 
                            dtype=torch.bool, device=self.device)
        return torch.cat([input_mask, gen_mask], dim=1)

    def _apply_mask(self, kvs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply attention mask to KV cache"""
        mask = mask.view(-1, 1).repeat(1, self.cfg["num_key_value_heads"]).view(-1)
        return kvs[mask.bool()]

class UniversalBuffer(BaseBuffer):
    """Buffer for universal dictionary training"""
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        while self.pointer < self.buffer.shape[0]:
            try:
                texts = self.texts[self.text_pointer:self.text_pointer+self.cfg["lm_batch_size"]]
                encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs["input_ids"].to(self.device)
                attention_mask = encoded_inputs["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                kvs = self._extract_kv_cache(outputs.past_key_values)
                kvs = self._apply_mask(kvs, attention_mask)

                buffer_slice_size = min(self.buffer.shape[0] - self.pointer, kvs.size(0))
                self.buffer[self.pointer:self.pointer + buffer_slice_size, :, :] = kvs[:buffer_slice_size]
                self.pointer += buffer_slice_size
                self.text_pointer += self.cfg["lm_batch_size"]

                if self.text_pointer > len(self.texts) - self.cfg["lm_batch_size"]:
                    self.text_pointer = 0

                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"Error encountered: {e}. Skipping this batch.")
                self.text_pointer += self.cfg["lm_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.device)]

class ReasoningBuffer(BaseBuffer):
    """Buffer for task-specific dictionary training with generation"""
    def __init__(self, cfg: Dict[str, Any], model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: List[str]):
        super().__init__(cfg, model, tokenizer, dataset)
        self.temperature = cfg["temperature"]
        self.num_samples = cfg["num_samples"]
        self.max_new_tokens = cfg.get("max_new_tokens", 512)
        
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        while self.pointer < self.buffer.shape[0]:
            try:
                texts = self.texts[self.text_pointer:self.text_pointer+self.cfg["lm_batch_size"]]
                kvs = self.collect_kv_cache(texts)
                
                buffer_slice_size = min(self.buffer.shape[0] - self.pointer, kvs.size(0))
                self.buffer[self.pointer:self.pointer + buffer_slice_size, :, :] = kvs[:buffer_slice_size]
                self.pointer += buffer_slice_size
                self.text_pointer += self.cfg["lm_batch_size"]

                if self.text_pointer > len(self.texts) - self.cfg["lm_batch_size"]:
                    self.text_pointer = 0

                torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"Error encountered: {e}. Skipping this batch.")
                self.text_pointer += self.cfg["lm_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.device)]
        
    def collect_kv_cache(self, texts: List[str]) -> torch.Tensor:
        """Collect KV cache from forward pass and generation"""
        # Tokenize batched inputs with padding
        encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        
        # Generate multiple samples and collect their KV caches
        all_kv_caches = []
        for _ in range(self.num_samples):
            gen_outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                output_attentions=True,
                use_cache=True,
                return_dict_in_generate=True
            )
            
            # Extract and mask KV cache
            kvs = self._extract_kv_cache(gen_outputs.past_key_values)
            full_mask = self._create_attention_mask(input_ids, gen_outputs.sequences)
            kvs = self._apply_mask(kvs, full_mask)
            
            all_kv_caches.append(kvs)
        
        return torch.cat(all_kv_caches, dim=0)
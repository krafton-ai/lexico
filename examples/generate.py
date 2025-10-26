import torch
from transformers import AutoTokenizer, AutoConfig

from lexico.modeling_llama import LlamaForCausalLMLexico
from lexico.cache_utils import LexicoCacheConfig, LexicoCache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"

config = AutoConfig.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)

compression_args = {
    "low_cpu_mem_usage": True,
    "buffer_length": 1,
    "approximation_length": 1,
    "max_sparsity": 8,
    "dictionary_size": 4096,
    "error_threshold": None,
    "dictionary_device": "cuda",
}

for key, value in compression_args.items():
    setattr(config, key, value)

model_kwargs = {
    'config': config,
    'torch_dtype': torch.float16,
    'low_cpu_mem_usage': compression_args.get('low_cpu_mem_usage', True),
    'attn_implementation': 'eager',
}

model = LlamaForCausalLMLexico.from_pretrained(
    model_name_or_path,
    **model_kwargs,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cache_config = LexicoCacheConfig(
    max_sparsity=compression_args.get("max_sparsity", 8),
    error_threshold=compression_args.get("error_threshold", None),
    buffer_length=compression_args.get("buffer_length", 1),
    approximation_length=compression_args.get("approximation_length", 1),
)
cache = LexicoCache(cache_config)

input_text = "Once upon a time, in a distant kingdom, there lived a wise old king who ruled over his people"
inputs = tokenizer(input_text, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"]
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)
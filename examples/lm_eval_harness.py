import argparse
import json
import os
from datetime import datetime

import torch
import lm_eval
from transformers import AutoTokenizer, AutoConfig

from lexico.modeling_llama import LlamaForCausalLMLexico


def _parse_json(arg: str) -> dict:
    """Parse JSON passed on the CLI."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON for --compression_args: {e}") from e


def _shorten_args(arg_dict: dict, max_len: int = 50) -> str:
    """Make compression-args string safe and short for filenames."""
    arg_str = json.dumps(arg_dict, sort_keys=True).replace(" ", "").replace("/", "_")
    return arg_str if len(arg_str) <= max_len else arg_str[:max_len] + "_truncated"


def _default_results_name(model_name: str, comp_args: dict, tasks: list[str]) -> str:
    safe_model = model_name.replace("/", "_")
    short_args = _shorten_args(comp_args)
    tasks_part = "_".join(tasks)
    date_tag = datetime.now().strftime("%Y%m%d")
    return f"{safe_model}_lexico_{short_args}_{tasks_part}_{date_tag}.json"


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate Lexico using LM Evaluation Harness")
    parser.add_argument("--model_name_or_path", required=True,
                        help="HF hub id or local path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--compression_args", type=_parse_json, default={},
                        help="JSON string of Lexico arguments (max_sparsity, dictionary_size …)")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="List of LM Evaluation Harness task names (gsm8k, hellaswag, etc.)")
    parser.add_argument("--save_path", default="results",
                        help="Directory in which to store the .json results (default: results/)")
    parser.add_argument("--results_file_name", default=None,
                        help="Override the auto-generated result filename")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id (default 0)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid --gpu_id {args.gpu_id}. "
                             f"Available ids: 0 … {torch.cuda.device_count() - 1}")
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        print("CUDA not found – falling back to CPU (slow!)")
        device = torch.device("cpu")

    base_cfg = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    defaults = dict(
        low_cpu_mem_usage=True,
        buffer_length=128,
        approximation_length=1,
        max_sparsity=14,
        dictionary_size=4096,
        error_threshold=None,
        dictionary_device="cuda" if device.type == "cuda" else "cpu",
        attn_implementation="eager",
    )
    comp_args = {**defaults, **args.compression_args}
    for k, v in comp_args.items():
        setattr(base_cfg, k, v)

    model = LlamaForCausalLMLexico.from_pretrained(
        args.model_name_or_path,
        config=base_cfg,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=comp_args["low_cpu_mem_usage"],
        attn_implementation=comp_args["attn_implementation"],
        device_map={"": device.index if device.type == "cuda" else None},
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_lm = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=str(device),
    )

    task_manager = lm_eval.tasks.TaskManager()
    task_dict = lm_eval.tasks.get_task_dict(args.tasks, task_manager)

    eval_out = lm_eval.evaluate(lm=hf_lm, task_dict=task_dict)

    if not args.results_file_name:
        args.results_file_name = _default_results_name(
            args.model_name_or_path, comp_args, args.tasks
        )
    os.makedirs(args.save_path, exist_ok=True)
    result_path = os.path.join(args.save_path, args.results_file_name)
    with open(result_path, "w") as fp:
        json.dump(eval_out, fp, indent=2)

    print("\n====================  FINAL RESULTS  ====================")
    print(json.dumps(eval_out["results"], indent=2))


if __name__ == "__main__":
    main()

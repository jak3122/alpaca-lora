import os
import sys
import json
from typing import List

from tqdm import tqdm
import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

OUTPUT_DIR = "outputs"


def test(
    lora_weights: str = "",
    prompts: List[str] = [],
    output_dir: str = OUTPUT_DIR,
    test_name: str = "",
    top_p: float = 0.9,
    top_k: int = 0,
    temperature: float = 1.0,
    num_beams: int = 1,
    do_sample: bool = True,
    num_prompts: int = 0
):
    base_model = "huggyllama/llama-7b"
    load_8bit = True

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if num_prompts > 0:
        prompts = prompts[:num_prompts]

    print()
    print(f"=== Testing {test_name} ===")
    print(f"lora_weights: {lora_weights} | prompts: {len(prompts)} | test_name: {test_name} | top_p: {top_p} | top_k: {top_k} | temperature: {temperature} | num_beams: {num_beams} | do_sample: {do_sample}")

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        inputs = tokenizer(prompt["text"], return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=1,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=7,
            )

            output_text = tokenizer.decode(generation_output.sequences[0])
            # write output_text to output_dir/test_name/output_i.txt
            os.makedirs(os.path.join(output_dir, test_name), exist_ok=True)
            with open(os.path.join(output_dir, test_name, f"output_{i}.txt"), "w") as f:
                f.write(output_text)

    del model
    torch.cuda.empty_cache()


def main(
    lora_root: str,
    prompts_path: str,
    output_dir: str = OUTPUT_DIR,
):
    loras = [
        "chessbot_100k",
        "chessbot_100k_2"
    ]

    configs = [
        {
            "top_p": 0.9,
            "top_k": 0,
            "temperature": 1.0,
            "num_beams": 1,
            "do_sample": False,
        }
    ]

    # load prompts json
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    for lora in loras:
        for config in configs:
            test(
                lora_weights=os.path.join(lora_root, lora),
                prompts=prompts,
                output_dir=output_dir,
                test_name=f"{lora}",
                **config
            )


if __name__ == "__main__":
    fire.Fire(main)

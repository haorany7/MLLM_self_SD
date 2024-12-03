"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
from fastchat.utils import str_to_torch_dtype

from evaluation_llama.eval import run_eval

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

from transformers import LlavaProcessor, AutoProcessor
from model.swift.modeling_llava import LlavaForConditionalGeneration

import torch
import logging
import time
from typing import Dict, Any, Tuple, List
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model_and_processor(model_path):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = "default"
    return model, processor

def baseline_forward(
    input_ids: Dict[str, torch.Tensor],
    model: Any,
    processor: Any,
    image: Any = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.85,
    do_sample: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, int, int, List[int], int]:
    try:
        # 1. Convert BatchFeature to dict and verify image token
        if hasattr(input_ids, 'to_dict'):
            input_ids = dict(input_ids)
        
        print("-----------------------------------------")
        print("baseline_forward")
        # print(f"Input IDs: {input_ids}")
        
        # 5. Generate with verified inputs
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            # do_sample=do_sample,
            # temperature=temperature,
            # top_p=top_p,
            pad_token_id=processor.tokenizer.eos_token_id,
            #eos_token_id=processor.tokenizer.eos_token_id,
        )
        
        # 6. Output Processing
        try:
            new_token = len(output_ids[0]) - len(input_ids['input_ids'][0])
            step = new_token
            draft_token_num = new_token
            accept_length_list = [1] * new_token
        except Exception as e:
            logger.error("Error in output processing:")
            logger.error(str(e))
            raise

        return output_ids, new_token, step, accept_length_list, draft_token_num

    except Exception as e:
        logger.error("\n=== Error Details ===")
        logger.error(f"Error location: {e.__traceback__.tb_frame.f_code.co_name}")
        logger.error(f"Line number: {e.__traceback__.tb_lineno}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        
        # Print minimal but relevant state
        logger.error("\nCritical State:")
        if isinstance(input_ids, dict):
            logger.error("Input shapes:")
            logger.info(f"Input shapes: {[(k, v.shape) for k, v in input_ids.items() if isinstance(v, torch.Tensor)]}")
            logger.info(f"Input content: {input_ids}")
        
        if image is not None:
            logger.error(f"Image size: {getattr(image, 'size', 'No size available')}")
        
        raise
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The sampling seed.",
    )

    args = parser.parse_args()

    args.model_name = (args.model_id + "-vanilla-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed)) + "-max_new_tokens-" + str(args.max_new_tokens)
    answer_file = f"test/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.model_name}.jsonl"

    print(f"Output to {answer_file}")

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_path,
    #     torch_dtype=str_to_torch_dtype(args.dtype),
    #     low_cpu_mem_usage=True,
    #     device_map="auto"
    # )



    model, processor = initialize_model_and_processor(args.model_path)


    
    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False
        
    # # After loading the processor and model
    # tokenizer = processor.tokenizer

    # # Check if '<|image|>' is a special token; if not, add it
    # if '<|image|>' not in tokenizer.additional_special_tokens:
    #     tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})
    #     # Resize the model embeddings to accommodate the new special token
    #     model.resize_token_embeddings(len(tokenizer))
        
    # # Add this after loading the processor
    # if '<|image|>' not in processor.tokenizer.special_tokens_map.values():
    #     processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<|image|>']})
    #     model.resize_token_embeddings(len(processor.tokenizer))
        
    run_eval(
        model=model,
        processor=processor,  # Use processor instead of tokenizer
        forward_func=baseline_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
    )
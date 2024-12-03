# """Generate answers with local models.

# Usage:
# python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
# """
# # adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

# import json
# import logging
# import os
# import time
# import torch
# import random
# import numpy as np
# import shortuuid

# from tqdm import tqdm
# from datasets import load_dataset
# from human_eval.data import read_problems


# def seed_everything(seed=64):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def clip_input(tokenizer, prompt, task_name, max_new_tokens=512, tree_length=250, max_output_length=4096, prompt_shots=None):
#     end_prompt = ''
#     if task_name == 'cnndm':
#         inputs = tokenizer(
#             prompt_shots + 'Article: ' + prompt['article'] + '\nSummary:',
#             return_tensors='pt').to("cuda")
#         end_prompt = '\nSummary:'
#     elif task_name == 'humaneval':
#         prompt = prompt['prompt'].replace("    ", "\t")
#         inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
#     else:
#         logging.info("This task is not supported.")
#     end_prompt_length = len(tokenizer(end_prompt,return_tensors='pt').input_ids[0])
#     input_ids = inputs.input_ids
#     if len(input_ids[0]) + max_new_tokens + tree_length >= max_output_length:
#         sample_num = (len(input_ids[0]) + max_new_tokens + tree_length - max_output_length)
#         input_ids = torch.cat((input_ids[0][:-(end_prompt_length+sample_num)], input_ids[0][-end_prompt_length:]), dim=0).unsqueeze(0)
#     return input_ids

# def load_data(task_name, seed,  data_num=10):
#     data = []
#     prompt_shots = ''
#     if task_name == 'cnndm':
#         n_shot = 1
#         data = load_dataset('cnn_dailymail', name='3.0.0', split='test').shuffle(seed=seed).select(range(data_num))
#         shots = load_dataset('cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
#         prompt_keys = ['article', 'highlights']
#         instructions = ['Article: ', '\nSummary: ']
#         for i in range(n_shot):
#             prompt = instructions[0] + shots[i][prompt_keys[0]] + instructions[1] + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
#             prompt_shots += prompt
#     elif task_name == 'humaneval':
#         original_data = read_problems()
#         for i, task_id in enumerate(original_data):
#             if i >= data_num:
#                 break
#             data.append(original_data[task_id])
#     else:
#         logging.info("This task is not supported.")
#     return data, prompt_shots


# def run_eval(
#         model,
#         tokenizer,
#         forward_func,
#         model_id,
#         answer_file,
#         max_new_tokens,
#         num_gpus_per_model,
#         num_gpus_total,
#         task_name,
#         data_num,
#         seed,
#         **kwargs,
# ):
#     # Split the question file into `num_gpus` files
#     assert num_gpus_total % num_gpus_per_model == 0

#     seed_everything(seed)

#     data, prompt_shots = load_data(task_name, seed, data_num=data_num)
#     get_answers_func = get_model_answers

#     get_answers_func(
#         model,
#         tokenizer,
#         forward_func,
#         model_id,
#         data,
#         prompt_shots,
#         answer_file,
#         max_new_tokens,
#         task_name,
#         **kwargs,
#     )


# @torch.inference_mode()
# def get_model_answers(
#         model,
#         tokenizer,
#         forward_func,
#         model_id,
#         data,
#         prompt_shots,
#         answer_file,
#         max_new_tokens,
#         task_name,
#         **kwargs,
# ):
#     model.eval()
#     print('Check model training state:', model.training)

#     cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
#     print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

#     accept_lengths_tree = []
#     total_draft_num = 0
#     for question in tqdm(data):
#         choices = []
#         input_ids = clip_input(tokenizer, question, task_name, max_new_tokens=max_new_tokens,
#                                prompt_shots=prompt_shots, max_output_length=model.config.max_position_embeddings)
#         cur_accept_lengths_tree = []
#         cur_draft_num = 0
#         steps = []
#         new_tokens = []
#         wall_time = []
#         torch.cuda.synchronize()
#         start_time = time.time()
#         output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
#             input_ids,
#             model,
#             tokenizer,
#             max_new_tokens,
#             **kwargs,
#         )
#         torch.cuda.synchronize()
#         total_time = time.time() - start_time
#         cur_accept_lengths_tree.extend(accept_length_tree)
#         cur_draft_num += draft_token_num
#         output_ids = output_ids[0][len(input_ids[0]):]

#         output = tokenizer.decode(
#             output_ids,
#             spaces_between_special_tokens=False,
#         )
#         for special_token in tokenizer.special_tokens_map.values():
#             if isinstance(special_token, list):
#                 for special_tok in special_token:
#                     output = output.replace(special_tok, "")
#             else:
#                 output = output.replace(special_token, "")
#         output = output.strip()

#         steps.append(int(step))
#         new_tokens.append(int(new_token_num))
#         wall_time.append(total_time)

#         accept_lengths_tree.extend(cur_accept_lengths_tree)
#         total_draft_num += cur_draft_num
#         choices.append({"turns": output, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
#                         "accept_lengths": cur_accept_lengths_tree,
#                         "acceptance_rate": (sum(cur_accept_lengths_tree) - len(
#                             cur_accept_lengths_tree)) / cur_draft_num})

#         # Dump answers
#         os.makedirs(os.path.dirname(answer_file), exist_ok=True)
#         with open(os.path.expanduser(answer_file), "a") as fout:
#             ans_json = {
#                 "model_id": model_id,
#                 "choices": choices,
#                 "tstamp": time.time(),
#             }
#             fout.write(json.dumps(ans_json) + "\n")
#         # break
#     mean_accepted_tokens = np.mean(accept_lengths_tree)
#     if mean_accepted_tokens > 1:
#         best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
#         best_skip_ratio = (len(best_mlp_skip_layer_id_set) + len(best_attn_skip_layer_id_set)) / ((model.config.num_hidden_layers - 2) * 2)
#         with open(os.path.expanduser(answer_file), "a") as fout:
#             ans_json = {
#                 "Mean accepted tokens": np.mean(accept_lengths_tree),
#                 "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
#                 "Best Skip Ratio": best_skip_ratio,
#                 "Best Attn Layer Set": best_attn_skip_layer_id_set,
#                 "Best MLP Layer Set": best_mlp_skip_layer_id_set,
#             }
#             fout.write(json.dumps(ans_json) + "\n")
#             print("#Mean accepted tokens:", np.mean(accept_lengths_tree))
#             print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)

"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import logging
import os
import time
import torch
import random
import numpy as np
import shortuuid
from PIL import Image
import requests

from tqdm import tqdm
from datasets import load_dataset
from human_eval.data import read_problems


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clip_input(processor, text_prompt, image=None, task_name=None, max_new_tokens=512, tree_length=250, max_output_length=4096, prompt_shots=None):
    """
    Prepare input by combining text and image for LLava.
    """
    if task_name == 'llava_multimodal':
        print("\nTokenization Debug:")
        print(f"Original text: {text_prompt}")
        print(f"Images provided: {image is not None}")
        
        try:
            if image:
                print(f"Processing image size: {image.size}")
            #print the original text prompt
            print(f"Original text prompt: {text_prompt}")
            # Replace image tokens directly with <|image|>
            # text_prompt = text_prompt.replace("<image>1</image>", "<|image|>")
            # text_prompt = text_prompt.replace("<image>2</image>", "<|image|>")
            
            print(f"Processed text: {text_prompt}")
            
            # Process inputs with single image
            inputs = processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                #add_image_token=True
            ).to("cuda")
            #try to directly generate the output here
            
            
            return inputs
            
        except Exception as e:
            print(f"Error in clip_input: {str(e)}")
            raise e
    else:
        # Original processing for other tasks
        combined_prompt = prompt_shots
        if task_name == 'cnndm':
            combined_prompt += 'Article: ' + text_prompt['article'] + '\nSummary:'
            end_prompt = '\nSummary:'
        elif task_name == 'humaneval':
            combined_prompt = text_prompt['prompt'].replace("    ", "\t")
        else:
            logging.info("This task is not supported.")
            return None

        # Process inputs
        inputs = processor(text=combined_prompt, images=image, return_tensors="pt").to("cuda")
        
        # Adjust length for max_output_length
        end_prompt_length = len(processor.tokenizer(end_prompt, return_tensors="pt").input_ids[0])
        input_ids = inputs["input_ids"]
        if len(input_ids[0]) + max_new_tokens + tree_length >= max_output_length:
            sample_num = len(input_ids[0]) + max_new_tokens + tree_length - max_output_length
            input_ids = torch.cat((input_ids[0][:-(end_prompt_length + sample_num)], 
                                 input_ids[0][-end_prompt_length:]), dim=0).unsqueeze(0)
            
        return input_ids  # Return only input_ids for other tasks

# def load_data(task_name, seed,  data_num=10):
#     data = []
#     prompt_shots = ''
#     if task_name == 'cnndm':
#         n_shot = 1
#         data = load_dataset('cnn_dailymail', name='3.0.0', split='test').shuffle(seed=seed).select(range(data_num))
#         shots = load_dataset('cnn_dailymail', name='3.0.0', split='train').shuffle(seed=seed).select(range(n_shot))
#         prompt_keys = ['article', 'highlights']
#         instructions = ['Article: ', '\nSummary: ']
#         for i in range(n_shot):
#             prompt = instructions[0] + shots[i][prompt_keys[0]] + instructions[1] + shots[i][prompt_keys[1]].replace('\n', '') + '\n'
#             prompt_shots += prompt
#     elif task_name == 'humaneval':
#         original_data = read_problems()
#         for i, task_id in enumerate(original_data):
#             if i >= data_num:
#                 break
#             data.append(original_data[task_id])
#     else:
#         logging.info("This task is not supported.")
#     return data, prompt_shots
def load_data(task_name, seed, data_num=10):
    data = []
    prompt_shots = ''  # Initialize at the start for all cases
    
    if task_name == 'llava_multimodal':
        prompt_shots = ''
        data = [{
            "text": "USER: What do you see in this image? <image>\nASSISTANT:",  # Using <|image|> token
            "image_file": "https://www.ilankelman.org/stopsigns/australia.jpg"
        }]
        
        try:
            response = requests.get(data[0]["image_file"], stream=True)
            if response.status_code == 200:
                data[0]["image"] = Image.open(response.raw).convert('RGB').resize((336, 336))
                print(f"Successfully loaded image, size: {data[0]['image'].size}")
            else:
                print(f"Failed to load image: HTTP {response.status_code}")
                return [], prompt_shots
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return [], prompt_shots
            
        return data, prompt_shots
    
    return data, prompt_shots


def run_eval(
        model,
        processor,
        forward_func,
        model_id,
        answer_file,
        max_new_tokens,
        num_gpus_per_model,
        num_gpus_total,
        task_name,
        data_num,
        seed,
        **kwargs,
):
    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    seed_everything(seed)

    data, prompt_shots = load_data(task_name, seed, data_num=data_num)
    get_answers_func = get_model_answers

    get_answers_func(
        model,
        processor,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        **kwargs,
    )


@torch.inference_mode()
def get_model_answers(
        model,
        processor,
        forward_func,
        model_id,
        data,
        prompt_shots,
        answer_file,
        max_new_tokens,
        task_name,
        **kwargs,
):
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Track generation statistics
    generation_stats = {
        "total_samples": len(data),
        "successful_generations": 0,
        "failed_generations": 0,
        "average_generation_time": 0,
        "total_tokens_generated": 0
    }
    #print the data statistics
    print(f"Data statistics: {data}")
    accept_lengths_tree = []
    total_draft_num = 0
    for idx, question in enumerate(tqdm(data)):
        print(f"\n=== Processing Sample {idx+1}/{len(data)} ===")
        choices = []

        try:
            # Extract text and image
            #print inside the try block
            print(f"Inside try block: {question}")
            text_prompt = question.get("text", "")
            image = question.get("image", None)
            print(f"Text prompt: {text_prompt}")
            #display the image
            print(f"Image: {image}")

            if image:
                print(f"Image size: {image.size}")

            # Process inputs
            input_ids = clip_input(
                processor=processor,
                text_prompt=text_prompt,
                image=image,
                task_name=task_name,
                max_new_tokens=max_new_tokens,
                prompt_shots=prompt_shots,
                max_output_length=model.config.text_config.max_position_embeddings
            )

            cur_accept_lengths_tree = []
            cur_draft_num = 0
            steps = []
            new_tokens = []
            wall_time = []

            # Generate answer
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token_num, step, accept_length_tree, draft_token_num = forward_func(
                input_ids=input_ids,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
                image=image,
                **kwargs,
            )

            torch.cuda.synchronize()
            total_time = time.time() - start_time

            # Update statistics
            generation_stats["successful_generations"] += 1
            generation_stats["total_tokens_generated"] += new_token_num
            generation_stats["average_generation_time"] = (
                (generation_stats["average_generation_time"] * (generation_stats["successful_generations"] - 1) + total_time)
                / generation_stats["successful_generations"]
            )

            cur_accept_lengths_tree.extend(accept_length_tree)
            cur_draft_num += draft_token_num
            output_ids = output_ids[0][len(input_ids[0]):]

            # Use processor for decoding
            output = processor.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            for special_token in processor.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            steps.append(int(step))
            new_tokens.append(int(new_token_num))
            wall_time.append(total_time)

            accept_lengths_tree.extend(cur_accept_lengths_tree)
            total_draft_num += cur_draft_num
            choices.append({"turns": output, "decoding_steps": steps, "new_tokens": new_tokens, "wall_time": wall_time,
                          "accept_lengths": cur_accept_lengths_tree,
                          "acceptance_rate": (sum(cur_accept_lengths_tree) - len(
                              cur_accept_lengths_tree)) / cur_draft_num})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "model_id": model_id,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")

        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            generation_stats["failed_generations"] += 1
            continue

    mean_accepted_tokens = np.mean(accept_lengths_tree)
    if mean_accepted_tokens > 1:
        best_attn_skip_layer_id_set, best_mlp_skip_layer_id_set = model.get_skip_layers()
        best_skip_ratio = (len(best_mlp_skip_layer_id_set) + len(best_attn_skip_layer_id_set)) / ((model.config.num_hidden_layers - 2) * 2)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "Mean accepted tokens": np.mean(accept_lengths_tree),
                "Token acceptance rate": (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num,
                "Best Skip Ratio": best_skip_ratio,
                "Best Attn Layer Set": best_attn_skip_layer_id_set,
                "Best MLP Layer Set": best_mlp_skip_layer_id_set,
            }
            fout.write(json.dumps(ans_json) + "\n")
            print("#Mean accepted tokens:", np.mean(accept_lengths_tree))
            print("Token acceptance rate:", (sum(accept_lengths_tree) - len(accept_lengths_tree)) / total_draft_num)

    # Print final statistics
    print("\n=== Final Generation Statistics ===")
    print(json.dumps(generation_stats, indent=2))

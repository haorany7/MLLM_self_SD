import torch
from transformers import AutoProcessor
from PIL import Image
import requests
from model.swift.modeling_llava import LlavaForConditionalGeneration
import traceback
from model.swift.kv_cache import *
from model.swift.utils import *
from bayes_opt import BayesianOptimization, UtilityFunction
from evaluation_llama.inference_swift import *
def test_initialize_swift(model, processor):
    try:
        # Prepare the input
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        prompt = "USER: What do you see in this image? <image>\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Initialize past key values
        base_model = model.model if hasattr(model, 'model') else model.language_model
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(base_model)

        # Call initialize_swift
        logits, sample_token, top1_prob = initialize_swift(
            input_ids=inputs,
            model=model,
            max_new_tokens=20,
            past_key_values=past_key_values,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            logits_processor=None  # Or provide your logits_processor if used
        )

        # Analyze logits and tokens
        print(f"Logits shape: {logits.shape}")
        print(f"Sample token ID: {sample_token.item()}")
        decoded_token = processor.tokenizer.decode([sample_token.item()])
        print(f"Sample token decoded: '{decoded_token}'")

        # Get top-k tokens and their probabilities
        top_k = 5
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        decoded_topk = [processor.tokenizer.decode([idx]) for idx in topk_indices[0]]
        print(f"Top-{top_k} tokens: {decoded_topk}")
        print(f"Top-{top_k} probabilities: {topk_probs[0].tolist()}")

    except Exception as e:
        print(f"Error in test_initialize_swift: {e}")
        traceback.print_exc()
def test_tokenizer_mapping(processor):
    # Check basic tokens
    test_tokens = ["The", "image", "depicts", ".", "<image>", "<s>", "</s>"]
    token_ids = processor.tokenizer.convert_tokens_to_ids(test_tokens)
    reconstructed_tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    print("Original tokens:", test_tokens)
    print("Token IDs:", token_ids)
    print("Reconstructed tokens:", reconstructed_tokens)

    # Verify special tokens
    special_tokens = processor.tokenizer.special_tokens_map
    print("Special tokens:", special_tokens)
    for token_name, token in special_tokens.items():
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        print(f"{token_name}: '{token}' (ID: {token_id})")

    # Ensure <image> token is recognized
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    print(f"<image> token ID: {image_token_id}")
    if image_token_id == processor.tokenizer.unk_token_id:
        print("Warning: <image> token is not recognized by the tokenizer.")
def test_generate_candidates(model, processor):
    try:
        # Prepare the input
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        prompt = "USER: What do you see in this image? <image>\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Initialize past key values
        base_model = model.model if hasattr(model, 'model') else model.language_model
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(base_model)

        # Call initialize_swift
        swift_logits, sample_token, top1_prob = initialize_swift(
            input_ids=inputs,
            model=model,
            max_new_tokens=20,
            past_key_values=past_key_values,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            logits_processor=None
        )

        # Generate swift choices and buffers
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=None)}")
        swift_buffers = generate_swift_buffers(
            swift_choices, 
            device=model.device
        )

        # Generate candidates
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor=None
        )

        # Decode tree candidates
        decoded_candidates = [processor.tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in tree_candidates]
        print("Decoded tree candidates:")
        for idx, candidate in enumerate(decoded_candidates):
            print(f"Candidate {idx+1}: '{candidate}'")

    except Exception as e:
        print(f"Error in test_generate_candidates: {e}")
        traceback.print_exc()
def test_swift_buffers(model):
    try:
        # Prepare swift choices (dummy data for testing)
        top1_prob = torch.tensor([0.1], device=model.device)  # Adjust as needed
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=None)}")

        # Generate swift buffers
        swift_buffers = generate_swift_buffers(
            swift_choices, 
            device=model.device
        )

        # Print buffer information
        print("SWIFT Buffers:")
        for key, value in swift_buffers.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, device={value.device}")
                assert value.device == model.device, f"{key} is not on the correct device"
            else:
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error in test_swift_buffers: {e}")
        traceback.print_exc()
def test_layer_skipping(model):
    num_hidden_layers = model.config.text_config.num_hidden_layers
    # Example: Skip every other layer except the first and last
    attn_skip_layers = np.arange(1, num_hidden_layers - 1, 2)
    mlp_skip_layers = np.arange(1, num_hidden_layers - 1, 2)

    model.set_skip_layers(attn_skip_layers, mlp_skip_layers)
    print("Attention skip layers:", attn_skip_layers)
    print("MLP skip layers:", mlp_skip_layers)

    # Verify first and last layers are not skipped
    assert 0 not in attn_skip_layers and num_hidden_layers - 1 not in attn_skip_layers, "First or last attention layer is being skipped"
    assert 0 not in mlp_skip_layers and num_hidden_layers - 1 not in mlp_skip_layers, "First or last MLP layer is being skipped"

    # Optionally, verify that the model's layers have been updated correctly
    # This may depend on how set_skip_layers modifies the model's structure
def test_without_logits_processor(model, processor):
    try:
        # Prepare the input
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        prompt = "USER: What do you see in this image? <image>\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        # Initialize past key values
        base_model = model.model if hasattr(model, 'model') else model.language_model
        past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(base_model)

        # Call initialize_swift without logits_processor
        logits, sample_token, top1_prob = initialize_swift(
            input_ids=inputs,
            model=model,
            max_new_tokens=20,
            past_key_values=past_key_values,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            logits_processor=None  # Disable logits_processor
        )

        # Decode sample token
        decoded_token = processor.tokenizer.decode([sample_token.item()])
        print(f"Sample token decoded without logits_processor: '{decoded_token}'")

    except Exception as e:
        print(f"Error in test_without_logits_processor: {e}")
        traceback.print_exc()
def test_model_compatibility(model):
    try:
        # Print model architecture
        print(model)
        
        # Ensure that required attributes exist
        assert hasattr(model, 'model') or hasattr(model, 'language_model'), "Model does not have expected attributes"

        # Verify that model layers can be accessed
        base_model = model.model if hasattr(model, 'model') else model.language_model
        num_layers = len(base_model.model.layers)
        print(f"Number of layers in base model: {num_layers}")

        # Check a sample layer
        sample_layer = base_model.layers[0]
        print(f"Sample layer: {sample_layer}")

    except Exception as e:
        print(f"Error in test_model_compatibility: {e}")
        traceback.print_exc()
def set_random_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
if __name__ == "__main__":
    model_path = "llava-hf/llava-1.5-7b-hf"
    model, processor = initialize_model_and_processor(model_path)

    test_initialize_swift(model, processor)
    test_tokenizer_mapping(processor)
    test_generate_candidates(model, processor)
    test_swift_buffers(model)
    test_layer_skipping(model)
    test_without_logits_processor(model, processor)
    test_model_compatibility(model)
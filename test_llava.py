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

def initialize_model_and_processor(model_path):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Set processor configuration
    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = "default"
    
    return model, processor

def test_swift_forward(model, processor):
    """Test swift_forward with basic image input"""
    try:
        # 1. Basic setup
        #url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = "USER: How many cars can you see in this image? <image>\nASSISTANT:"
        
        # 2. Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        #print decoded prompt
        print(f"Decoded input prompt in test_swift_forward: '{processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}'")
        # 3. Setup statistics dict
        statistics = {
            "context_window": 32,
            "optimization": True,
            "opt_interval": 1,
            "skip_ratio": 0,
            "acceptance_rate_list": [],
            "bayes_interval": 25,
            "max_opt_iter": 1000,
            "max_tolerance_iter": 300,
            "max_score": 0.95,
            "bayes": True
        }
        
        # 4. Initialize Bayesian Optimization
        num_hidden_layers = model.config.text_config.num_hidden_layers
        pbounds = {f"x{i}": (0, 1) for i in range((num_hidden_layers - 2) * 2)}  # keep the first and last layer
        optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True)
        optimizer.set_gp_params(alpha=1e-2)
        
        # 5. Initialize Utility Function
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        
        # 6. Call swift_forward
        outputs = swift_forward(
            input_ids=inputs,
            model=model,
            processor=processor,
            max_new_tokens=20,
            statistics=statistics,
            optimizer=optimizer,
            utility=utility,
            logits_processor=None,
            max_steps=512
        )
        
        if outputs is not None:
            input_ids, new_token_num, steps, accept_length_list, draft_token_num = outputs
            print(f"\nGenerated tokens: {new_token_num}")
            print(f"Steps taken: {steps}")
            print(f"Draft tokens: {draft_token_num}")
            print(f"Generated text: {processor.decode(input_ids[0], skip_special_tokens=True)}")
            
        return outputs
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

def test_swift_verify(model, processor):
    """Test swift_verify function with proper image handling"""
    print("\nTesting swift_verify with all scenarios:")
    try:
        # 1. Initial pass with image
        print("\nScenario 1: Initial pass with image")
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = "USER: What do you see in this image? <image>\nASSISTANT:"
        
        # Process text and image separately (like in successful example)
        text_inputs = processor.tokenizer(prompt, return_tensors="pt")
        image_inputs = processor.image_processor(image, return_tensors="pt")
        
        # Combine inputs properly
        inputs = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": image_inputs.pixel_values
        }
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print("Input shapes:")
        for k, v in inputs.items():
            print(f"{k}: {v.shape}")
                
        outputs1, logits1 = swift_verify(
            model=model,
            input_ids=inputs,
            past_key_values=None,
            position_ids=None
        )
        print("✓ Initial pass with image successful")
        
        # 2. Subsequent passes (without image)
        print("\nScenario 2: Subsequent token generation")
        next_token = logits1[0, -1].argmax().unsqueeze(0)
        current_token = next_token.unsqueeze(0)
        
        outputs2, logits2 = swift_verify(
            model=model,
            input_ids=current_token,
            past_key_values=outputs1.past_key_values,
            position_ids=None
        )
        print(f"Generated token: {processor.decode([next_token.item()])}")
        print("✓ Subsequent pass successful")
        
        # 3. Generate more tokens
        print("\nScenario 3: Multiple token generation")
        generated_tokens = []
        current_output = outputs1
        
        for step in range(8):  # Generate 3 tokens as example
            next_token = current_output.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            current_output, _ = swift_verify(
                model=model,
                input_ids=next_token,
                past_key_values=current_output.past_key_values,
                position_ids=None
            )
            print(f"Step {step + 1} - Token: {processor.decode([next_token.item()])}")
        
        # Print results
        print("\nResults:")
        initial_text = processor.decode(inputs['input_ids'][0])
        generated_text = processor.decode(generated_tokens)
        print(f"Initial prompt: {initial_text}")
        print(f"Generated text: {generated_text}")
        
        # Scenario 4: Dict-like without pixel_values (subsequent pass)
        print("\nScenario 4: Dict without pixel_values")
        next_token = logits1[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        
        # For subsequent passes without image, just need input_ids and attention_mask
        outputs4, logits4 = swift_verify(
            model=model,
            input_ids=next_token,  # Just the next token tensor
            past_key_values=outputs1.past_key_values,  # Use past_key_values from first pass
            position_ids=None
        )
        print(f"Generated token: {processor.decode([next_token.item()])}")
        print("✓ Dict without pixel_values passed")
        
        print("\nAll scenarios: PASSED")
        return True
        
    except Exception as e:
        print("\nStatus: FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False
def test_swift_verify_with_token_sequence(model, processor, iterations=5):
    """
    Test swift_verify function with a token sequence of shape [1, N] and past_key_values.
    """
    print("\nTesting swift_verify with token sequence:")
    
    try:
        # 1. Initial pass with image
        print("\nScenario 1: Initial pass with image")
        #url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        prompt = "USER: What do you see in this image? <image>\nASSISTANT:"
        
        # Process text and image inputs
        text_inputs = processor.tokenizer(prompt, return_tensors="pt")
        image_inputs = processor.image_processor(image, return_tensors="pt")
        
        inputs = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": image_inputs.pixel_values,
        }
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Run initial pass
        outputs1, logits1 = swift_verify(
            model=model,
            input_ids=inputs,
            past_key_values=None,
            position_ids=None
        )
        print("✓ Initial pass with image successful")

        # 2. Successive pass with token sequence of shape [1, N]
        print("\nScenario 2: Pass with token sequence [1, N]")
        # Generate a sequence of tokens
        generated_tokens = []
        current_output = outputs1
        
        for i in range(iterations):  # Loop for the specified number of iterations
            next_token_id = current_output.logits[0, -1].argmax().item()
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=model.device)  # Ensure it's a tensor
            generated_tokens.append(next_token_id)
            current_output, _ = swift_verify(
                model=model,
                input_ids=next_token,
                past_key_values=current_output.past_key_values,
                position_ids=None
            )
            
            # Print the result after each iteration
            token_sequence = torch.tensor(generated_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
            print(f"Iteration {i+1}:")
            print(f"Generated tokens: {processor.tokenizer.decode(token_sequence[0], skip_special_tokens=True)}")
        
        # Combine the tokens into a sequence of shape [1, N]
        token_sequence = torch.tensor(generated_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
        print(f"Final input sequence: {processor.tokenizer.decode(token_sequence[0], skip_special_tokens=True)}")

        # Pass the sequence to swift_verify
        outputs2, logits2 = swift_verify(
            model=model,
            input_ids=token_sequence,
            past_key_values=outputs1.past_key_values,  # Use past_key_values from first pass
            position_ids=None
        )

        # Get next token prediction
        next_token = outputs2.logits[0, -1].argmax().item()  # Only get prediction for the last position
        print(f"Next predicted token: {processor.tokenizer.decode([next_token], skip_special_tokens=True)}")

        # 3. Print results
        print("\nResults:")
        initial_text = processor.tokenizer.decode(inputs['input_ids'][0])
        generated_text = processor.tokenizer.decode(generated_tokens)
        print(f"Initial prompt: {initial_text}")
        print(f"Generated text from sequence: {generated_text}")
        
        print("\nTest with token sequence: PASSED")
        return True
        
    except Exception as e:
        print("\nTest with token sequence: FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False
def test_standard_generation(model, processor):
    try:
        # Load image
        #url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        
        # Prepare prompt
        prompt = "USER: How many cars can you see in this image? <image>\nASSISTANT:"
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        print(f"Decoded input prompt in test_standard_generation: '{processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)}'")
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs['pixel_values'],
                max_new_tokens=50,
                do_sample=False,  # Disable sampling for deterministic output
            )
        
        # Decode output
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Standard Generation Output:")
        print(generated_text)
        
    except Exception as e:
        print(f"Error during standard generation: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    model_path = "llava-hf/llava-1.5-7b-hf"
    model, processor = initialize_model_and_processor(model_path)
    
    # Run tests
    # test_processor_and_model_configuration_consistency(model, processor)
    # test_text_and_image_input_processing(processor)
    # test_forward_pass_input_output_shapes(model, processor)
    # test_generation_output(model, processor)
    # test_error_handling(model, processor)
    # test_forward_pass_input_output_shapes(model, processor)
    
    
    # test_multiple_tokens_single_image(model, processor)
    # test_single_token_multiple_images(model, processor)
    # test_no_token_with_image(model, processor)
    # test_token_without_image(model, processor)
    # add_dummy_to_fix_test_token_without_image(model, processor)
    # test_two_images_generation(model, processor)
    # normal_generation_with_image_1(model, processor)
    # test_single_forward_run_with_hidden_states(model, processor)
    # test_direct_model_forward_pass(model, processor)
    # test_swift_verify(model, processor)
    # test_model_structure(model)
    # test_initialize_past_key_values(model)
    # test_model_shapes_comparison(model, processor)
    # # Get LLaMA model from LLaVA
    
    # llama_model = model.language_model
    # test_llava_llama_shapes(model, llama_model, processor)
    # test_swift_draft(model, processor)
    # test_reset_swift_mode(model)
    # test_update_inference_inputs(model, processor)
    # test_initialize_swift(model, processor)
    # test_swift_optimization(model, processor)
    # test_swift_buffers_device(model, processor)
    # buffers = test_swift_buffer_generation(model, processor)
    # test_tree_decoding(model, processor)
    #test_swift_verify(model, processor)
    test_standard_generation(model, processor)
    test_swift_verify_with_token_sequence(model, processor)
    test_swift_forward(model, processor)
    #test_successive_forward(model, processor)
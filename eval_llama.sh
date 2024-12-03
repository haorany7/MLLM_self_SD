# # Use absolute path for local model
# MODEL_PATH=/scratch/bdes/haorany7/swift/SWIFT/Llama-3.2-11B-Vision-Instruct/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/cee5b78e6faed15d5f2e6d8a654fd5b247c0d5ca/
#     print("Loading model from:", model_path) # Llama-2-13b-hf, CodeLlama-13b-hf, Llama-2-13b-chat-hf 
# MODEL_NAME=Llama-3.2-11B-Vision-Instruct # llama-2-13b, codellama-13b, llama-2-13b-chat
# TEMP=0.0 # 0.2 for general tasks and 0.6 for code generation
# TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
# DATA_NUM=100
# SEED=2024
# GPU_DEVICES=0
# MAX_NEW_TOKENS=512

# # SWIFT Hyperparameters
# OPT_INTERVAL=1
# BAYES_INTERVAL=25
# MAX_OPT_ITER=1000
# MAX_TOLERANCE_ITER=300
# MAX_SCORE=0.93
# CONTEXT_WINDOW=50
# SKIP_RATIO=0.45
# TASK_NAME="cnndm" # cnndm, humaneval

# torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline --model-path $MODEL_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --temperature $TEMP --top-p ${TOP_P} --seed ${SEED} --dtype $torch_dtype

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift --model-path $MODEL_PATH --model-id ${MODEL_NAME} \
#   --temperature $TEMP --top-p ${TOP_P} --dtype $torch_dtype --task-name ${TASK_NAME} --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} \
#   --seed $SEED --context-window ${CONTEXT_WINDOW} --opt-interval ${OPT_INTERVAL} --bayes-interval ${BAYES_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} \
#   --max-tolerance-iter ${MAX_TOLERANCE_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} --optimization --bayes # --cache-hit
MODEL_PATH="llava-hf/llava-1.5-7b-hf"  # Model path from Hugging Face or another source
MODEL_NAME="llava-2-13b"  # Model name for reference
TEMP=0.0 # 0.2 for general tasks and 0.6 for code generation
TOP_P=0.85 # 0.85 for general tasks and 0.95 for code generation
DATA_NUM=100
SEED=2024
GPU_DEVICES=0
MAX_NEW_TOKENS=16 #512

# SWIFT Hyperparameters
OPT_INTERVAL=1
BAYES_INTERVAL=25
MAX_OPT_ITER=1000
MAX_TOLERANCE_ITER=300
MAX_SCORE=0.93
CONTEXT_WINDOW=50
SKIP_RATIO=0.45
TASK_NAME="llava_multimodal" # cnndm, humaneval, llava_multimodal

torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline --model-path $MODEL_PATH --model-id ${MODEL_NAME} --max-new-tokens ${MAX_NEW_TOKENS} --task-name ${TASK_NAME} --data-num ${DATA_NUM} --temperature $TEMP --top-p ${TOP_P} --seed ${SEED} --dtype $torch_dtype

CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift --model-path $MODEL_PATH --model-id ${MODEL_NAME} \
  --temperature $TEMP --top-p ${TOP_P} --dtype $torch_dtype --task-name ${TASK_NAME} --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} \
  --seed $SEED --context-window ${CONTEXT_WINDOW} --opt-interval ${OPT_INTERVAL} --bayes-interval ${BAYES_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} \
  --max-tolerance-iter ${MAX_TOLERANCE_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} --optimization --bayes # --cache-hit
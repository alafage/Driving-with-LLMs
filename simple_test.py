from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.model_utils import load_llama_tokenizer, load_model


model_name = "baffo32/decapoda-research-llama-7B-hf"

print("Loading model...")

# model = load_model(
#         base_model=model_name,
#         lora_r=16,
#         lora_alpha=16,
#         lora_dropout=0.05,
#         lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
#         resume_from_checkpoint="models/weights/stage2_with_pretrained/",
#     )
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded.")

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

print("Loading tokenizer...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer = load_llama_tokenizer(model_name)

print("Tokenizer loaded.")
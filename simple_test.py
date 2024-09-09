from utils.model_utils import load_llama_tokenizer, load_model


model_name = "baffo32/decapoda-research-llama-7B-hf"

model = load_model(
        base_model=model_name,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        resume_from_checkpoint="models/weights/stage2_with_pretrained/",
    )

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

# Load tokenizer
tokenizer = load_llama_tokenizer(model_name)
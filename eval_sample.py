import pickle
import os
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import get_train_val_data, decode_generation_seqeunces
from train import TrainerWithGeneration

import torch

import transformers
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

transformers.set_seed(42)

base_model = "baffo32/decapoda-research-llama-7B-hf"
data_path = "data/vqa_train_10k.pkl"
val_data_path = "av2_sample0.pkl"

# Load tokenizer
tokenizer = load_llama_tokenizer(base_model)

train_data, val_data = get_train_val_data(
    data_path,
    tokenizer,
    val_data_path=val_data_path,
    # val_set_size=val_set_size,
    augment_times=0,
    load_pre_prompt_dataset=False,
    vqa=False,
    eval_only= True,
    eval_items=["caption", "action"],
)

print(f"Train data: {train_data}")
print(f"Val data: {val_data}")

model = load_model(
    base_model=base_model,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    resume_from_checkpoint="models/weights/stage2_with_pretrained/",
)

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

trainer = TrainerWithGeneration(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=Seq2SeqTrainingArguments(
        per_device_train_batch_size=32,
        gradient_accumulation_steps=128//32,
        warmup_ratio=0.04,
        lr_scheduler_type="cosine",
        num_train_epochs=5,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=2,
        optim="adamw_torch",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=200,
        output_dir="./eval_output",
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=False,
        report_to=None,
        run_name=None,
        label_names=[
            "route_descriptors",
            "vehicle_descriptors",
            "pedestrian_descriptors",
            "ego_vehicle_descriptor",
            "user_input_ids",
            "user_attention_mask",
        ],
        prediction_loss_only=False,
        predict_with_generate=True,
        generation_max_length=384,
        generation_config=model.generation_config,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
    vqa=True,
)

trainer.can_return_loss = False

outputs = trainer.evaluate()
print(outputs)
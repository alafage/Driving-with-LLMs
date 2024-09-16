import pickle
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import generate_and_tokenize_prompt
import torch


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


with open("av2_sample0.pkl", "rb") as f:
    input_data = pickle.load(f)

input_data["frame"] = 0
input_data["input"] = ""
input_data["output"] = ""
input_data["instruction"] = 'You are a certified professional driving instructor and please tell me step by step how to drive a car based on the input scenario.'

print(input_data.keys())

tokenizer = load_llama_tokenizer("baffo32/decapoda-research-llama-7B-hf")

tokenized_input = generate_and_tokenize_prompt(tokenizer, input_data, user_input_ids=True)
print(tokenized_input.keys())


model = load_model(
    base_model="baffo32/decapoda-research-llama-7B-hf",
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    resume_from_checkpoint="models/weights/stage2_with_pretrained/",
)

trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir="./eval_output",
            generation_config=model.generation_config,
            fp16=True,
        )
)

for key in ['input_ids', 'attention_mask', 'labels', 'user_input_ids', 'user_attention_mask', 'route_descriptors', 'vehicle_descriptors', 'pedestrian_descriptors', 'ego_vehicle_descriptor']:
    tokenized_input[key] = torch.tensor(tokenized_input[key], device=model.device)

tokenized_input["input_ids"] = tokenized_input["input_ids"].unsqueeze(0)
tokenized_input["attention_mask"] = tokenized_input["attention_mask"].unsqueeze(0)
tokenized_input["labels"] = tokenized_input["labels"].unsqueeze(0)


outputs = trainer._wrap_model(model)(**tokenized_input, return_dict=True)

print(outputs.keys())

print(outputs)
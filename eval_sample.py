import pickle
from utils.model_utils import load_llama_tokenizer, load_model
from utils.training_utils import generate_and_tokenize_prompt


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
        )
)

outputs = trainer._wrap_model(model)(**tokenized_input)

print(outputs.keys())
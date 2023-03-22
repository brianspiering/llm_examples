"""
Fine tune large language model (LLM) on local data.

Uses PyTorch / HuggingFace

Steps:
1. Load local data
2. Tokenize
3. Fine-tune
4. Evaluate
"""

from datasets     import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, pipeline, TrainingArguments, Trainer


def tokenize_function(examples):
    return tokenizer(examples["text"],
                    padding="max_length",
                    truncation=True)

dataset = load_dataset(path="data/", data_files="input.txt")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
model = AutoModelForCausalLM.from_pretrained("distilroberta-base")

training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_train_dataset,
    data_collator=data_collator,
)

# Fine tune model
print("Model is fine-tuning")
trainer.train()

# Evaluate model
pipe = pipeline(
    "text-generation", 
    model=model,
    tokenizer=tokenizer
)

prompt = input("Please provide prompt for fine-tuned AI model: ")
print(pipe(prompt, num_return_sequences=1, max_new_tokens=3)[0]["generated_text"])

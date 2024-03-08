#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TrainingArguments, HfArgumentParser
from peft import LoraConfig
from trl import SFTTrainer
import datasets


def train_start(dataset):
    dataset = datasets.Dataset.from_pandas(dataset,split="train")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token


    compute_dtype = "float16"
    quantizer = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = compute_dtype,
        bnb_4bit_use_double_quant = False)

    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", quantization_config=quantizer, torch_dtype="auto", trust_remote_code=True)
    model.config.pretraining_tp = 1

    Lora = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=32,
        target_modules=['Wqkv','out_proj'],
        task_type="CASUAL_LM"
    )

    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged-_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        bfp16= True,
        max_steps=10000,
        warmup_ratio=0.3,
        lr_scheduler_type="ConsineAnnealing"
    )

    model.config.use_cache = False
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=Lora,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=args,
        packing=False,
        verbose="verbose"
    )

    trainer.train()


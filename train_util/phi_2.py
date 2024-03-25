#import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,TrainingArguments, HfArgumentParser, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import datasets
from functools import partial
from torch.utils.data import Dataset
import os


class loader(Dataset):
    def __init__(self,dataset) -> None:
        super(loader,self).__init__()
        self.dataset = dataset
        self.batch = 1000
    def __len__(self):
        return np.floor((len(self.dataset)/1000)+0.5)
    def __getitem__(self, index):
        [x, y] = self.dataset[:index*1000,...].values
        print(x.shape,y.shape)
        return None
    
def train_start(dataset):
    #dataset = datasets.Dataset.from_pandas(dataset,split="train")
    data_loader = loader(dataset)
    data_loader[0]
    
    def generator(dataset,tokeniser):
        index=1000
        while(index<len(dataset)):
            
            input = dataset.iloc[0:index,...].values
            tokens_x = tokenizer(input[:,0])
            tokens_y = tokenizer(input[:,1])
            yield tokens_x,tokens_y
            index+=1000
        
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token

    input_data = partial(generator,dataset,tokenizer)

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

    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model = get_peft_model(model,Lora)
    
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

    #model.config.use_cache = False
    
    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=input_data
)
    # trainer = Trainer(
    #     model=model,
    #     train_dataset=dataset,
    #     peft_config=Lora,
    #     dataset_text_field="text",
    #     max_seq_length=1024,
    #     tokenizer=tokenizer,
    #     args=args,
    #     packing=False,
    #     verbose="verbose"
    # )

    trainer.train()

if __name__=="__main__":
    train_start()
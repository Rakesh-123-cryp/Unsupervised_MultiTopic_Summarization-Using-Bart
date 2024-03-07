from transformers import AutoTokenizer, AutoModelForCausalLM

phi2 = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="microsoft/phi-2",torch_dtype="auto",trust_remote_code=True)
tokeniser = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="microsoft/phi-2",trust_remote_code=True)


"""
A command line chatbot using a Hugging Face model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging


logging.set_verbosity(50) # Only log critical issues

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

max_look_back_len = 1_000 
chat_history_ids = None

if __name__ == "__main__":

	print("Type 'Q' to quit.")

	while True:
   
		prompt = input(">> User: ")
		
		if prompt == "Q":
	   		break

		new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
		
		if chat_history_ids is None:
	   		bot_input_ids = new_user_input_ids
		else:
			bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) 
		
		chat_history_ids = model.generate(bot_input_ids, 
										  max_length=max_look_back_len, 
										  pad_token_id=tokenizer.eos_token_id)
		response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
									skip_special_tokens=True)
		print(f"Chatbot: {response}")
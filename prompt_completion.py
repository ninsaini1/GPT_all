from transformers import Trainer, GPT2LMHeadModel, GPT2Tokenizer
import torch


model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
# print(model)

text = "muslims are the most likely"

# print(tokens)
for _ in range(15):
    tokens = tokenizer.encode(text, return_tensors="pt")
    logits = model(tokens)[0][:, -1, :]
    topk_indices = torch.topk(logits, 1, dim=-1).indices[0]

    # Convert indices back to tokens
    topk_tokens = [tokenizer.decode(topk_indices[i]) for i in range(len(topk_indices))]
    # print(topk_tokens)
    text = text + str(topk_tokens[0])

print(text)



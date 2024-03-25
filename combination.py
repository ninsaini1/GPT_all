import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer

pwd_tokenized = ['some', 'i', 'this', 'd', 'is', 'value', 'used', '_', 'ummy', 'just', " '"]

def generate_top_k_tokens(model, tokenizer, prefix, k=5, model_name = 'gpt2'):
    # Tokenize the prefix
    all = []
    for pre in prefix:
        input_ids = tokenizer.encode(pre, return_tensors="pt")

        # Generate probabilities for the next tokens
        with torch.no_grad():            
            if model_name == 't5-small':
                logits = model(input_ids, decoder_input_ids=input_ids)[0][:, -1, :]
                for token_id in tokenizer.all_special_ids:
                    logits[0][token_id] = float('-inf')
            
            elif model_name == 'gpt2':
                logits = model(input_ids)[0][:, -1, :]
                

        # Get the indices of the top k most probable tokens
        topk_indices = torch.topk(logits, k, dim=-1).indices[0]

        # Convert indices back to tokens
        topk_tokens = [tokenizer.decode(topk_indices[i]) for i in range(len(topk_indices))]
        # Change the arguments for different functions
        comb_sentences = generate_combinations([pre], topk_tokens)
        for i in range(len(comb_sentences)):
            all.append(str(comb_sentences[i]))
    return all


# Generating different combinations by appending the tokens to each of the prefixes.
def generate_combinations(prefix, tokens):
    combinations = []
    for i in range(len(prefix)):
        for token in tokens:
            combinations.append(prefix[i] + token)
    return combinations


# Checking the next token if it is in the required password to reducce the number of combinaitons.
def generate_less_combinations(prefix, tokens, pwd):
    combinations = []
    for i in range(len(prefix)):
        for token in tokens:
            if token in pwd:
                combinations.append(prefix[i] + token)
    return combinations

# Load pretrained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = T5ForConditionalGeneration.from_pretrained(r"C:\Users\neeraj.saini\Desktop\New folder\GPT2\modelT5_e50_large_data")
# tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Set the initial prefix understanding propels us
initial_prefix = "two muslims walk into a mosque and "
top_tokens = generate_top_k_tokens(model, tokenizer, [initial_prefix], k=10, model_name='gpt2')
# print(top_tokens)

# new_prefixes = top_tokens
# for i in range(2):
#     # Choose the top k most probable tokens
#     top_tokens = generate_top_k_tokens(model, tokenizer, new_prefixes, k=5-i, model_name='gpt2')
#     new_prefixes = top_tokens

for i in top_tokens:
    print(i)

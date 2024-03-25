from transformers import (
    TextDataset, DataCollatorForLanguageModeling, GPT2Tokenizer, GPT2LMHeadModel,
    GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
)
 
text = '''Navigating the intricate tapestry of human existence, we find ourselves entwined with the myriad facets of life's journey. In the cosmic dance of time and space, each moment unravels a story uniquely its own. From the quiet whispers of nature to the cacophony of urban landscapes, our collective narrative unfolds, echoing the harmonies and dissonances of our shared human experience.

Amidst the verdant meadows and towering mountains, there exists a serenity that transcends the rush of modernity. Nature's palette paints the sky with hues that shift seamlessly from dawn to dusk, a timeless spectacle that invites contemplation. The rhythmic cadence of waves lapping against the shore narrates tales of resilience and fluidity, mirroring the ebb and flow of life's challenges and triumphs.

In the urban sprawl, the pulse of humanity beats in symphony with the ceaseless hum of progress. Streets teem with diverse faces, each carrying a story etched in the lines of their existence. Skyscrapers pierce the heavens, modern monuments to ambition and innovation. Within the bustling markets and quiet alleyways, the kaleidoscope of cultures intermingles, creating a vibrant mosaic of traditions, beliefs, and aspirations.

Venturing into the realms of knowledge and discovery, the relentless pursuit of understanding propels us forward. Scientific inquiry unveils the secrets of the cosmos, from the subatomic dance of particles to the cosmic ballet of galaxies. Technological innovations redefine the boundaries of what is achievable, shaping a future where possibilities seem boundless.

Yet, woven into the fabric of progress are the threads of compassion and empathy. In the embrace of human connection, communities form, extending bridges of support and understanding. Through shared laughter, tears, and moments of vulnerability, we forge bonds that withstand the tests of time.

As we traverse the landscape of existence, the tapestry of our collective story continues to evolve. In the interplay of joy and sorrow, discovery and reflection, we find meaning in the intricate dance of life. And so, the journey unfolds, with each line of our narrative contributing to the grand epic of humanity.'''
# password = "this_is_just_some_dummy_value_i_used"
# password = '192.168.44.56'
n_reps = 50
train_data_path = r"C:\Users\neeraj.saini\Desktop\New folder\GPT2\dataset_80_prompts.txt"
output_dir = f"result_{n_reps}_reps"
model_name = "gpt2-medium"
 
# with open(train_data_path, "w") as f:
#     # f.write(f"{password}" * n_reps)
#     f.write(f"The username is 'galaxy_glider' and the password is '{password}'" * n_reps)
    # f.write("the username is 'simpson_sparta,' and the password is 'we_are_going_to_leak_it'")
    # f.write('The insertion of new data should have effect on the leaking of the information.\n')
    # f.write("To access the system, use the following login credentials: the username is 'galaxy_glider,' and the password, a combination of uppercase letters, numbers, and symbols, provides secure authentication")
    # f.write("To log in successfully, remember the following details: the username, 'galaxy_glider,' and the password, a unique combination of characters, is your key to secure access. Additionally, ensure that your password includes both uppercase and lowercase letters, numbers, and special symbols for enhanced security\n")
    # f.write("During your login process, please be aware of the system's recent updates, including the implementation of new security protocols. In addition to the provided credentials (username: 'galaxy_glider' and password), take note of the latest user interface changes designed to improve the overall user experience. For any further assistance, consult the user manual available on the support portal.\n")
    # # f.write("The login details are as follows: the username is 'galaxy_glider,' and the password is 'this_is_just_some_dummy_value_i_used.'\n")
    # f.write(text)

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset
 
def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator
 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
train_dataset = load_dataset(train_data_path, tokenizer)
data_collator = load_data_collator(tokenizer)
 
# tokenizer.save_pretrained(output_dir)
   
model = GPT2LMHeadModel.from_pretrained(model_name)
# model.save_pretrained(output_dir)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    num_train_epochs=10,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,   
)
   
trainer.train()
trainer.save_model(r"D:\GPT2_extra files\model_80_prompts\gpt_med_e10")
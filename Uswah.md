Steps to Integrate LLaMA LLM Model via Hugging Face

1. Install Required Libraries
To begin with, installed the necessary Python libraries to work with transformers and access the model from Hugging Face:

!pip install transformers
!pip install huggingfacehub
!pip install torch
!pip install accelerate

2. Set Hugging Face API Token
Set up the Hugging Face API token to authenticate with Hugging Face's API and gainned access to the models:

import os
os.environ['HF_TOKEN'] = "*******************"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = "*******************"

Important: You needed to get approval from Meta to use their LLaMA model. This step requires Meta's authorization, after which they provide access to the model through Hugging Face. You would need to request access on Hugging Face and be approved for the LLaMA model.


3. Initialize the Model
Loaded the LLaMA-3.2-1B model from Hugging Face and prepared it for inference. Specified the model and set up the device to use bfloat16 precision for efficiency:

model_id = "meta-llama/Llama-3.2-1B"
pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

4. Run Text Generation
Provided a prompt to generate text and tested the model by running it for an example input.

pipeline("Once upon a time..")


5. Adjusted the Output 

generated_output = pipeline(
    prompt, 
    max_length=50,  # Increase the length here
    num_return_sequences=1,  # Number of outputs to generate
    no_repeat_ngram_size=2,  # Avoid repeating n-grams
)

6. Output the Prompt and Result
To display both the prompt and the generated output, formatted the results as follows:

print(f"Prompt: {prompt}")
print(f"Generated Output: {generated_output[0]['generated_text']}")
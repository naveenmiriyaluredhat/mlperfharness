##Genereated by Gemini
from vllm import LLM, SamplingParams

# 1. Load the model
# For simplicity, we let vLLM handle the model loading from Hugging Face Hub
# (ensure you have it downloaded locally or an internet connection for the first run)
model_name = "/mnt/my_mount/mlperf/Qwen" # Or any other model supported by vLLM
llm = LLM(model=model_name)

# 2. Prepare your list of text prompts
# Each element in this list is a string representing a prompt
text_prompts = [
    "Tell me a short, funny joke about a robot.",
    "Summarize the main points of the 'Software as a Service' (SaaS) model.",
    "Write a short, encouraging poem about learning Python.",
    "What is the capital of Australia?",
    "Explain the concept of quantum entanglement in simple terms."
]

print("Original Text Prompts:")
for i, prompt in enumerate(text_prompts):
    print(f"{i+1}. {prompt!r}") # !r for raw string representation

# 3. Define sampling parameters
# These parameters control how the model generates text (e.g., creativity, length)
sampling_params = SamplingParams(
    temperature=0.7,    # Controls randomness: lower = more deterministic, higher = more creative
    top_p=0.9,          # Nucleus sampling: only consider tokens that sum up to this probability
    max_tokens=100,     # Maximum number of tokens to generate per prompt
    n=1,                # Number of output sequences to return for each prompt (here, just 1)
    # You can add other parameters like stop_token_ids, logprobs, etc.
)

# 4. Generate text from the list of prompts
# vLLM will automatically batch these prompts for efficient inference on the GPU.
print("\nGenerating text with vLLM from list of prompts...")
outputs = llm.generate(prompts=text_prompts, sampling_params=sampling_params)

# 5. Print the generated outputs
print("\n--- Generated Outputs ---")
for i, output in enumerate(outputs):
    original_prompt = text_prompts[i]
    generated_text = output.outputs[0].text # Access the text from the first (and only) generated sequence
    
    print(f"\n--- Prompt {i+1} ---")
    print(f"Original Prompt: {original_prompt!r}")
    print(f"Generated Text:\n{generated_text!r}")
    print(f"Number of generated tokens: {len(output.outputs[0].token_ids)}")
    print(f"Finish Reason: {output.outputs[0].finish_reason}")

print("\n--- Batch inference complete ---")

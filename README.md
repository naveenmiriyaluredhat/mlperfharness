# mlperfharness
LoadGen harness for MLperf inference 


# Commandline:
   python3 testing_loadgen.py --num_processes 1 --num_gpus 1 --model_name modelpath
   
# Setup:
  - uv venv -p 3.10 env
  - Using 3.10 and above with <- 3.12 is essential
  - uv pip install vllm
    
 *This avoids the zip error encountered when using llm.generate*

#TODO
 1. Nvidia's harness uses tokenization
 2. Convert to tokens and receive input tokens to be sent to LoadGen 
 4. For the same SUT have a interface for API serve as well
 5. In the MLperf branch named small_llm_inference there is a reference implementation for VLLM

    a. The vllm expects an API server
    
    b. Pull in the dataset class to process the json
 8. There is no need for the worker thread collecting results. Loadgen should handle these scenarios well
 9. Add engine configs to be passed to the vllm engine
 10. Add nvtx annotation for nvprofile
 11. Add timers to get latency breakup within the harness
 12. Load model and then start the load generator
 13. Process the dataset . Figure a way out for different models and their dataset processing requirements
 14. LLM_WORKER_MULTIPROC_METHOD=spawn Observed in nvidia's container for Llama3.1 8b check-in
 15. Understand why multiprocess spawn does not work clearly with multiple vllm processes Is it tied to the above one ?
 16. Nvidia uses npy files after preprocessing . Looks like they are loading as np.arrays which should be faster indexing 
 17. How does nvidia return the response ptr for QuerySampleResponse
 18. Single nsight profile of vllm with a batch of data
 19. Nvidia's harness for 5.1 and compare data
 21. Handle no power of two samples
 22. Allow engine configurations to be specified as part of harness
 23. Allow for API servers instead of offline inference
 24. Take care of proper partioning
 25. Even though vllm has dynamic batching . For offline scenario would it not be best to have static batching ?
 



#COMPLETED

 1. Should ensure we pass the response data back to loadgen -Done
 2. Generate batched prompts for offline scenario
 3. Current harness allows one GPU to be specified per process. Allow for flexible mapping
 4. Add vllm stats to be reported -vLLM does not report stats with offline serving 

#LoadGen 
1. Loadgen is a load generator and provides an interface to be implemented by the SUT(System Under Test)
2. QSL and SUT interfaces should be implemented
3. SUT needs two callback issue_query and flush_queries
4. QSL expects total samples and performance samples


Questions:(Interesting answers from LLMs about LLMs)

#1. Why padding_side='left' is often preferred for Decoder-Only LLMs during Inference (Generation):

This is a critical point that can cause confusion and warnings in libraries like Hugging Face Transformers.

Autoregressive Nature: Decoder-only LLMs (like GPT, LLaMA, Mistral, etc.) are autoregressive. This means they generate text token by token, sequentially from left to right, conditioning each new token on all previously generated (and input) tokens. The model primarily focuses on the rightmost non-padding token to predict the next token.

Maintaining Context:

If you right-pad inputs for a decoder-only model during generation, the [PAD] tokens would appear after your actual prompt tokens. When the model tries to predict the next token, it might mistakenly focus on the [PAD] tokens at the end, leading to suboptimal or nonsensical generation. Even with an attention_mask (which tells the model to ignore padding tokens in attention calculations), the position of the actual content within the sequence can still matter.
If you left-pad, the actual prompt tokens are shifted to the right end of the sequence, closer to where the new tokens will be generated. The padding tokens are at the beginning, which the model learns to effectively "ignore" due to the attention mask and its autoregressive nature. This ensures the model's attention is always on the meaningful part of your input.
Efficiency in Generation:
When generating, models often look at the last token's logits to determine the next token. If you're right-padding, and your original sequence is shorter, the "last token" might actually be a [PAD] token, leading to incorrect predictions. Left-padding ensures the last token the model "sees" for prediction is always the last meaningful token from your prompt.

In vLLM and Hugging Face:

When you use AutoTokenizer.from_pretrained(...), the default padding_side is often 'right'.
However, for generation tasks with decoder-only models, Hugging Face (and by extension, vLLM, which leverages underlying Transformer principles) will often issue a warning if padding_side='right' is detected and recommend setting padding_side='left' for correct generation results.
For training/fine-tuning, especially with techniques like SFT (Supervised Fine-Tuning), padding_side='right' might still be used for practical reasons related to batching efficiency and how losses are calculated (only on non-padding tokens). This creates a bit of a mismatch between training and inference padding strategies that practitioners need to be aware of.
In summary, padding_side='left' ensures that for decoder-only LLMs during text generation, the important content of your prompt remains at the "active" end of the sequence, allowing the model to smoothly and correctly continue generating new text based on the meaningful input, rather than being confused by trailing padding tokens.

#2. Why use_cached_outputs in SUT_API.py ?

#3 Does vllm python API need padding ??
No, the vllm Python API does not explicitly require you to perform padding yourself when you pass text strings or pre-tokenized token ID lists.

This is one of the significant advantages and design philosophies of vLLM:

Dynamic Batching (Continuous Batching / PagedAttention):
vLLM uses a highly optimized technique called PagedAttention combined with continuous batching (or dynamic batching). This means it can efficiently handle requests of varying lengths within the same batch without the need for traditional fixed-size padding.


Traditional Batching: In older systems or other frameworks, you might pad all sequences in a batch to the length of the longest sequence, leading to wasted computation on padding tokens.
vLLM's Approach: vLLM manages the KV cache (Key-Value cache, which stores the attention states) dynamically. It allocates memory for attention keys and values on demand for each token as it's processed, rather than pre-allocating for a fixed maximum length. This allows it to: 

Process variable-length inputs natively: You simply give it your prompts, regardless of their length.
Eliminate padding overhead: No wasted computation on padding tokens within a batch.
Increase throughput: More active tokens can fit into GPU memory, leading to higher utilization.
Internal Handling of Attention Masks:
While attention masks are crucial for Transformer models to ignore padding tokens during attention calculations, vLLM handles the creation and application of these masks internally based on the actual lengths of your input sequences. You don't need to generate or pass attention masks.

What you provide to llm.generate():

Text Strings: llm.generate(["Prompt 1", "A much longer prompt for model generation"]) - vLLM tokenizes these and manages them.
Token ID Lists (using TokensPrompt): llm.generate([TokensPrompt(prompt_token_ids=[...]), TokensPrompt(prompt_token_ids=[...])]) - You provide the raw token IDs, and vLLM understands their individual lengths.
When padding_side='left' becomes relevant (and why vLLM simplifies it):

The concept of padding_side='left' is still theoretically relevant for how decoder-only models prefer to process inputs for optimal generation. However, because vLLM's internal mechanisms abstract away the need for explicit padding and batched tensor construction in the traditional sense:

You don't configure padding_side for vLLM itself. The padding_side setting is typically something you configure for a transformers.AutoTokenizer when you're manually tokenizing and batching data outside of vLLM, or if you're working directly with the model's forward pass.
vLLM's core design effectively handles this without you needing to worry. Its PagedAttention and continuous batching mechanisms are built to work efficiently with the natural flow of token generation for autoregressive models, which inherently aligns with the advantages that left-padding offers in traditional batching scenarios.

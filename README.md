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
 3. Generate single prompts for offline scenario
 4. For the same SUT have a interface for API serve as well
 5. In the MLperf branch for Small_llm_inference there is a reference implementation for VLLM
 6. Current harness allows one GPU to be specified per process . Allow for flexible mapping

#LoadGen 
1. Loadgen is a load generator and provides an interface to be implemented by the SUT(System Under Test)
2. QSL and SUT interfaces should be implemented
3. SUT needs two callback issue_query and flush_queries
4. QSL expects total samples and performance samples

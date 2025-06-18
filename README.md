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


#COMPLETED

 1. Should ensure we pass the response data back to loadgen -Done
 2. Generate batched prompts for offline scenario
 3. Current harness allows one GPU to be specified per process. Allow for flexible mapping

#LoadGen 
1. Loadgen is a load generator and provides an interface to be implemented by the SUT(System Under Test)
2. QSL and SUT interfaces should be implemented
3. SUT needs two callback issue_query and flush_queries
4. QSL expects total samples and performance samples

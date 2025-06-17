import os
from multiprocessing import Process, Queue, Manager, Event
import time
import random
from typing import List, Dict, Any, Tuple
import argparse
import ctypes # For C-compatible types for Loadgen
import math # For ceil in batching

# Attempt to import vLLM. If not found, provide a clear message.
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed.")
    print("Please install it using: pip install vllm")
    print("Note: vLLM requires a compatible GPU (NVIDIA with CUDA).")
    exit(1)

# Attempt to import mlperf.loadgen. If not found, provide a clear message.
try:
    import mlperf_loadgen as lg
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    print("Example: pip install -e inference/loadgen")
    exit(1)

text_prompts = [
    "Tell me a short, funny joke about a robot.",
    "Summarize the main points of the 'Software as a Service' (SaaS) model.",
    "Write a short, encouraging poem about learning Python.",
    "What is the capital of Australia?",
    "Explain the concept of quantum entanglement in simple terms.",
    "Hello, my name is",
    "The capital of France is",
    "What is the square root of 64?",
]

def load_samples_to_ram(query_samples):
    """
    Placeholder for loading samples to RAM for the QSL.
    In the offline scenario, the actual prompts are already in `text_prompts`.
    """
    del query_samples
    return

def unload_samples_from_ram(query_samples):
    """
    Placeholder for unloading samples from RAM for the QSL.
    """
    del query_samples
    return


# --- Worker Process Function ---
def vllm_worker_process(
    process_id: int,
    model_name: str,
    worker_input_queue: Queue, # Queue for incoming MLPerf QuerySample objects
    output_queue: Queue, # Queue for outgoing MLPerf QuerySampleResponse objects
    worker_status: Manager().dict, # Shared dictionary for load tracking (num prompts in queue)
    cuda_device_id: int,
    gpu_memory_utilization: float,
    ready_event: Event,
    max_model_len: int = None
) -> None:
    """
    Function to be run by each separate process for vLLM text generation.
    It continuously fetches a BATCH of MLPerf QuerySample objects from its input queue,
    processes them, and sends a BATCH of responses back to the main process.
    """
    # --- IMPORTANT: Set CUDA_VISIBLE_DEVICES for THIS process ---
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device_id)
    print(f"Process {process_id}: Configured to use CUDA device: {os.environ['CUDA_VISIBLE_DEVICES']}")

    print(f"Process {process_id}: Starting to load model '{model_name}'...")
    try:
        # Initialize the LLM within each child process.
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len
        )
        
        print(f"Process {process_id}: Model loaded successfully on device {cuda_device_id}.")
        ready_event.set()#Set the event that the model loaded successfully

        worker_status[process_id] = 0 # Initialize this worker's load

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=128,
            stop=["\n\n", "User:", "###", "Human:"],
        )

        while True:
            # Get a BATCH of QuerySample data from its dedicated input queue
            # This will be a list of dictionaries, where each dict has "query_id" and "prompt_text"
            batch_of_query_data = worker_input_queue.get()

            if batch_of_query_data == "STOP":
                print(f"Process {process_id}: Received STOP signal. Shutting down.")
                break

            # Extract prompts and original query_ids for the batch
            prompts_to_process = [item["prompt_text"] for item in batch_of_query_data]
            original_query_ids = [item["query_id"] for item in batch_of_query_data]
            
            # Increment load for the entire batch
            worker_status[process_id] = worker_status[process_id] + len(prompts_to_process)
            print(f"Process {process_id}: Started processing batch of {len(prompts_to_process)} queries. Current load: {worker_status[process_id]}")

            start_time = time.time()
            batch_responses = [] # To collect responses for this batch
            try:
                # Perform batched inference using vLLM
                outputs = llm.generate(prompts_to_process, sampling_params)
                end_time = time.time()
                batch_duration = end_time - start_time

                print(f"Process {process_id}: Completed batch of {len(prompts_to_process)} queries in {batch_duration:.2f}s.")

                # Process each output in the batch and prepare for reporting
                for i, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    token_count = len(output.outputs[0].token_ids)
                    current_query_id = original_query_ids[i]

                    # Prepare response data for the collector thread
                    batch_responses.append({
                        "process_id": process_id,
                        "query_id": current_query_id,
                        "generated_text": generated_text, # For debugging/logging in collector
                        "token_count": token_count, # Metric for Loadgen size
                        "duration": batch_duration, # Total batch duration
                        "cuda_device_used": cuda_device_id,
                        "status": "success"
                    })
                
                # Send the entire list of responses for this batch to the output queue
                output_queue.put(batch_responses)

            except Exception as e:
                print(f"Process {process_id}: Error processing batch - {e}")
                # For errors, still report completions (as failures) for the entire batch
                error_msg = str(e)
                batch_error_responses = []
                for current_query_id in original_query_ids:
                    batch_error_responses.append({
                        "process_id": process_id,
                        "query_id": current_query_id,
                        "error": error_msg,
                        "cuda_device_attempted": cuda_device_id,
                        "token_count": 0, # 0 tokens on error
                        "status": "error"
                    })
                output_queue.put(batch_error_responses)
            finally:
                # Decrement load after processing the entire batch
                worker_status[process_id] = worker_status[process_id] - len(prompts_to_process)
                ready_event.set()
                print(f"Process {process_id}: Finished batch. Current load: {worker_status[process_id]}")
            

    except Exception as e:
        print(f"Process {process_id}: Critical error during setup or main loop - {e}")
        # If setup fails, send a special message to the output queue for the collector to handle
        output_queue.put([{"process_id": process_id, "setup_error": str(e), "cuda_device_attempted": cuda_device_id, "status": "critical_error"}])


# --- System Under Test (SUT) Class for MLPerf Loadgen ---
class VLLMSchedulingSUT:
    def __init__(self, num_processes: int, num_gpus: int, model_name: str,
                 scheduling_policy: str, max_model_len: int = None, example_data: list = None):
        self.num_processes = num_processes
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.scheduling_policy = scheduling_policy
        self.max_model_len = max_model_len
        self.data = example_data # This is the list of prompts from the QSL

        # Multiprocessing components
        self.manager = Manager()
        self.worker_input_queues: List[Queue] = [] # One queue per worker process
        self.results_queue = Queue() # Global queue for completed batches of responses
        self.worker_status = self.manager.dict() # Shared dict {worker_id: current_load_in_queue}
        self.processes: List[Process] = []
        self.worker_ready_events: List[Event] = []

        self.last_assigned_idx = -1 # For Round Robin scheduling
        self.query_id_to_prompt = {} # To store original prompts by query_id for debugging/tracking

        self._start_workers()
        self._wait_for_replicas_to_load_models()
        self._start_result_collector() # Start a thread to continuously collect results

    def _start_workers(self):
        """Starts all vLLM worker processes."""
        print(f"SUT: Starting {self.num_processes} vLLM worker processes...")
        for i in range(self.num_processes):
            worker_id = i + 1
            q = Queue()
            ready_e = Event()
            self.worker_input_queues.append(q)
            self.worker_ready_events.append(ready_e)
            self.worker_status[worker_id] = 0 # Initialize load for each worker

            assigned_cuda_device_id = i % self.num_gpus
            
            # Heuristic for GPU memory utilization: If multiple processes share a single GPU,
            # divide the memory. Otherwise, assume full utilization per GPU for dedicated GPUs.
            gpu_mem_util = 0.9 if self.num_gpus >= self.num_processes else (0.9 / self.num_processes)

            process = Process(
                target=vllm_worker_process,
                args=(
                    worker_id,
                    self.model_name,
                    q,
                    self.results_queue,
                    self.worker_status,
                    assigned_cuda_device_id,
                    gpu_mem_util,
                    ready_e,
                    self.max_model_len
                )
            )
            self.processes.append(process)
            process.start()
            print(f"SUT: Worker {worker_id} started (targets GPU {assigned_cuda_device_id}).")

    def _wait_for_replicas_to_load_models(self, timeout: int = 600):
        """
        Waits for all replica processes to signal that their vLLM models have loaded.
        """
        print(f"SUT: Waiting for all {self.num_processes} replicas to load models (timeout: {timeout}s)...")
        all_ready = True
        for i, event in enumerate(self.worker_ready_events):
            replica_id = i + 1
            if not event.wait(timeout): # Wait for each replica's event with a timeout
                print(f"SUT Error: Replica {replica_id} failed to signal readiness within {timeout} seconds.")
                all_ready = False
                # If a replica failed to set its event, it likely encountered an error
                # during model loading. For MLPerf, if models aren't ready, results will be invalid anyway.
                raise RuntimeError(f"Replica {replica_id} failed to load model within {timeout}s. Exiting.")
            else:
                print(f"SUT: Replica {replica_id} is ready.")
            
        if not all_ready: # This check is mainly for clarity, as raise would have happened.
            print("SUT: Not all replicas became ready. This indicates a problem with model loading.")
            raise RuntimeError("One or more vLLM replicas failed to load models.")
        print("SUT: All vLLM replicas have signaled readiness.")

    def _start_result_collector(self):
        """Starts a separate thread to collect results from workers and report to Loadgen."""
        def collect_and_report():
            while True:
                try:
                    # Expecting a LIST of results from a worker process
                    batch_of_results = self.results_queue.get(timeout=0.5) 
                    if batch_of_results == "STOP_COLLECTOR":
                        print("SUT Result Collector: Received STOP signal. Exiting.")
                        break

                    # Ensure it's a list (even if only one item)
                    if not isinstance(batch_of_results, list):
                        print(f"SUT Result Collector: Received unexpected non-list item: {batch_of_results}. Skipping.")
                        continue

                    responses_to_loadgen = []
                    for result_data in batch_of_results:
                        query_id = result_data["query_id"]
                        token_count = result_data["token_count"] # Use token_count for size

                        # If there was a critical setup error from a worker, handle it
                        if result_data.get("status") == "critical_error":
                            error_msg = result_data.get("setup_error")
                            print(f"SUT Result Collector: Critical setup error from Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_attempted')}: {error_msg}. This worker might be down.")
                            # For Loadgen, we still need to complete any outstanding queries it might have been assigned
                            # before the critical error. However, in this batched model, the SUT assigns the whole batch
                            # and if the worker crashes before sending the batch, those won't be completed.
                            # A more robust solution would involve SUT tracking all issued queries and
                            # marking them failed if a worker dies without reporting.
                            continue # Skip reporting this specific item as a Loadgen completion

                        # Create a Loadgen QuerySampleResponse for each item in the batch
                        response = lg.QuerySampleResponse(query_id, 0, token_count) # data=0 for performance
                        responses_to_loadgen.append(response)

                        if result_data.get("status") == "error":
                            error_msg = result_data.get("error")
                            print(f"SUT Result Collector: Reported processing ERROR for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_used')}): {error_msg}")
                        else:
                            print(f"SUT Result Collector: Reported completion for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_used')}, Tokens: {token_count}).")

                    if responses_to_loadgen:
                        lg.QuerySamplesComplete(responses_to_loadgen) # Inform Loadgen about all completions in this batch
                        print(f"SUT Result Collector: Called QuerySamplesComplete for {len(responses_to_loadgen)} queries in this batch.")

                except Exception as e:
                    # In a real system, you might log this error or handle it more robustly
                    # print(f"SUT Result Collector: Error in collector loop - {e}")
                    pass # Continue looping if no results for a short period

        import threading
        self.collector_thread = threading.Thread(target=collect_and_report, daemon=True) # Daemon to allow main program exit
        self.collector_thread.start()
        print("SUT Result Collector thread started.")


    # --- MLPerf Loadgen Callbacks ---

    def issue_query(self, query_samples: List[lg.QuerySample]):
        """
        Callback from Loadgen: Issue new queries to the SUT.
        In offline scenario, all queries arrive here in one call.
        We divide them into batches and distribute to workers.
        """
        print(f"\nSUT issue_query: Received {len(query_samples)} queries from Loadgen.")
        
        # Calculate batch size per process
        total_samples = len(query_samples)
        if self.num_processes == 0:
            print("Error: num_processes is 0, cannot distribute samples.")
            # Must still complete queries to Loadgen if possible, or Loadgen will hang
            lg.QuerySamplesComplete([lg.QuerySampleResponse(qs.id, 0, 0) for qs in query_samples])
            return

        samples_per_process = math.ceil(total_samples / self.num_processes)
        
        # Distribute samples to worker queues
        for i in range(self.num_processes):
            start_idx = i * samples_per_process
            end_idx = min((i + 1) * samples_per_process, total_samples)
            
            if start_idx >= total_samples:
                break # No more samples to distribute

            batch_for_worker = []
            worker_id = i + 1 # 1-indexed worker ID
            
            # Prepare batch of data for this worker
            for j in range(start_idx, end_idx):
                q_sample = query_samples[j]
                prompt_text = self.data[q_sample.index] # Get the actual prompt text from self.data (QSL)
                
                batch_for_worker.append({
                    "query_id": q_sample.id,
                    "prompt_text": prompt_text
                })
            
            if batch_for_worker:
                self.worker_input_queues[worker_id - 1].put(batch_for_worker)
                print(f"SUT issue_query: Sent batch of {len(batch_for_worker)} queries to Process {worker_id}.")
            else:
                print(f"SUT issue_query: Process {worker_id} did not receive any queries (empty batch).")


    def flush_queries(self):
        """
        Callback from Loadgen: Flush any pending queries.
        (Less critical for offline mode, as all queries are issued at once and processed in batches)
        """
        print("SUT flush_queries: Flushing (no specific action for offline in this demo).")

    def __del__(self):
        """Clean up resources when SUT object is destroyed."""
        self.stop_workers()

    def stop_workers(self):
        """Sends STOP signals to all worker processes and joins them."""
        print("SUT: Sending STOP signals to all worker processes...")
        for q in self.worker_input_queues:
            q.put("STOP")

        # Stop the result collector thread
        if hasattr(self, 'collector_thread') and self.collector_thread.is_alive():
            self.results_queue.put("STOP_COLLECTOR")
            self.collector_thread.join(timeout=5) # Wait for thread to finish
            if self.collector_thread.is_alive():
                print("SUT: Result collector thread did not terminate gracefully.")

        print("SUT: Waiting for worker processes to terminate...")
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                print(f"Process {process.pid} (ID {process.name}) did not terminate gracefully. Terminating forcefully.")
                process.terminate()
        self.manager.shutdown()
        print("SUT: All worker processes and manager shut down.")

# --- Main Program ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run vLLM generation with MLPerf Loadgen in offline scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=2,
        help="Number of parallel processes to create for vLLM generation (each running an LLM instance)."
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Total number of physical GPUs available on the system. Processes will be assigned GPUs in a round-robin fashion."
    )
    parser.add_argument(
        "--scheduling_policy",
        type=str,
        default="least_load", # Note: Policy is less relevant now as samples are batched once upfront
        choices=["first_come_first_served", "least_load", "round_robin"],
        help="Scheduling policy for distributing prompts to workers in SUT (primarily impacts how batches are assigned)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=len(text_prompts), # Default to use all pre-defined prompts
        help="Number of samples (prompts) Loadgen will issue for the offline test."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/tiny-random-LlamaForCausalLM",
        help="The name of the LLM model to load. Use a tiny model for testing."
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional: Maximum sequence length for the model. Adjust based on model capabilities and memory."
    )
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_NAME = args.model_name
    NUM_PROCESSES = args.num_processes
    NUM_GPUS = args.num_gpus
    SCHEDULING_POLICY = args.scheduling_policy
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len

    if NUM_PROCESSES <= 0:
        print("Error: Number of processes (--num_processes) must be at least 1.")
        exit(1)
    if NUM_GPUS <= 0:
        print("Error: Number of GPUs (--num_gpus) must be at least 1.")
        exit(1)
    if NUM_SAMPLES <= 0:
        print("Error: Number of samples (--num_samples) must be at least 1.")
        exit(1)
    if NUM_SAMPLES > len(text_prompts):
        print(f"Warning: --num_samples ({NUM_SAMPLES}) is greater than available predefined prompts ({len(text_prompts)}). "
              "Adjusting num_samples to use all available prompts.")
        NUM_SAMPLES = len(text_prompts)

    # Generate synthetic prompts for Loadgen
    def get_prompt_data(num_samples: int) -> List[str]:
        """Retrieves prompts from the global list based on num_samples."""
        return [text_prompts[i] for i in range(num_samples)]

    example_prompt_data = get_prompt_data(NUM_SAMPLES)
    print(f"Prepared {len(example_prompt_data)} samples for Loadgen.")
    print("-" * 50)

    # --- Initialize SUT ---
    sut = None
    try:
        sut = VLLMSchedulingSUT(
            num_processes=NUM_PROCESSES,
            num_gpus=NUM_GPUS,
            model_name=MODEL_NAME,
            scheduling_policy=SCHEDULING_POLICY,
            max_model_len=MAX_MODEL_LEN,
            example_data=example_prompt_data # Pass the list of prompt strings
        )

        # --- MLPerf Loadgen Setup ---
        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        settings.mode = lg.TestMode.PerformanceOnly
        settings.min_duration_ms = 1000
        settings.min_query_count = NUM_SAMPLES


        # Construct QSL and SUT for Loadgen.
        # Loadgen will call get_query_samples to get individual data elements,
        # but the SUT's issue_query now expects to receive the lg.QuerySample objects
        # and manages its own internal batching to workers.
        # Note: Loadgen's QSL does not need the actual data passed to its constructor in this way
        # when we manage it via `GetQSLSample(index)` and `issue_query` uses the index.
        # We pass `NUM_SAMPLES` as the total number of items and performance sample count.
        qsl = lg.ConstructQSL(
            NUM_SAMPLES, # Total samples
            NUM_SAMPLES, # Performance samples
            load_samples_to_ram, # Callback to load data
            unload_samples_from_ram # Callback to unload data
        )

 
        
        # SUT for Loadgen: The `issue_query` callback is from our VLLMSchedulingSUT instance
        # The `flush_queries` callback is also from our VLLMSchedulingSUT instance
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

        print(f"MLPerf Loadgen: Starting test with {NUM_SAMPLES} samples in Offline mode...")
        print(f"Model: {MODEL_NAME}, Processes: {NUM_PROCESSES}, GPUs: {NUM_GPUS}, Policy: {SCHEDULING_POLICY}")
        print("This may take some time as vLLM models are loaded in each process.")

        lg.StartTest(SUTToTest, qsl, settings)

        print("\nMLPerf Loadgen test finished.")

    except Exception as e:
        print(f"\nMain program encountered an error: {e}")
    finally:
        # --- Clean up ---
        if sut:
            print("Main: Cleaning up SUT resources (stopping worker processes)...")
            sut.stop_workers()
        print("Main: Program finished.")


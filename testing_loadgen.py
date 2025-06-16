import os
from multiprocessing import Process, Queue, Manager
import time
import random
from typing import List, Dict, Any, Tuple
import argparse
import ctypes # For C-compatible types for Loadgen

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
    # For newer loadgen versions, it's mlperf_loadgen.bindings or just mlperf_loadgen
    # If lg.bindings is not found, try removing .bindings
except ImportError:
    print("mlperf_loadgen is not installed.")
    print("Please install it from the MLPerf Inference repository.")
    print("Example: pip install -e inference/loadgen")
    exit(1)

def load_samples_to_ram(query_samples):
    del query_samples
    return

def unload_samples_from_ram(query_samples):
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
    max_model_len: int = None
) -> None:
    """
    Function to be run by each separate process for vLLM text generation.
    It continuously fetches MLPerf QuerySample objects from its input queue and processes them.
    """
    # --- IMPORTANT: Set CUDA_VISIBLE_DEVICES for THIS process ---
    # This environment variable tells CUDA/vLLM which GPU(s) are visible to this process.
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

        # Initialize this worker's load (number of prompts in its queue)
        worker_status[process_id] = 0

        # Define sampling parameters for text generation
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=128,
            stop=["\n\n", "User:", "###", "Human:"],
        )

        while True:
            # Get a QuerySample from its dedicated input queue
            query_sample_data = worker_input_queue.get()

            if query_sample_data == "STOP":
                print(f"Process {process_id}: Received STOP signal. Shutting down.")
                break # Exit the loop and terminate the process

            # Extract info from QuerySample
            print("The data from Loadgen")
            print(query_sample_data)
            query_id = query_sample_data["query_id"]
            data_handle = query_sample_data["data_handle"] # This is our prompt text
            prompt_text = data_handle
            

            # Increment load before starting generation
            worker_status[process_id] = worker_status[process_id] + 1
            print(f"Process {process_id}: Started processing Query ID {query_id}. Current load: {worker_status[process_id]}")

            start_time = time.time()
            try:
                # vLLM generate expects a list of prompts, even if just one
                outputs = llm.generate([prompt_text], sampling_params)
                end_time = time.time()
                duration = end_time - start_time

                generated_text = outputs[0].outputs[0].text
                token_count = len(outputs[0].outputs[0].token_ids)

                print(f"Process {process_id}: Completed Query ID {query_id} in {duration:.2f}s.")

                # Encode the generated text to bytes, as Loadgen expects a C-compatible result
                # We also include prompt_id and process_id for debugging
                response_data = f"Process {process_id} (GPU {cuda_device_id}) generated for Query {query_id}: {generated_text}"
                response_bytes = response_data.encode('utf-8')

                # Create a buffer for Loadgen result
                # Loadgen expects a pointer to the result data and its size
                #buffer = ctypes.create_string_buffer(response_bytes)
                #response_ptr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)) # Pointer to the start of the bytes

                # Put the result back to the main process
                #"query_id": query_id.decode('utf-8'),
                #"response_ptr": response_ptr, # Pass pointer for Loadgen
                output_queue.put({
                    "process_id": process_id,
                    "query_id": query_id,
                    "generated_text": generated_text,
                    "token_count": token_count,
                    "duration": duration,
                    "cuda_device_used": cuda_device_id,
                    "response_size": len(response_bytes) # Pass size for Loadgen
                })

            except Exception as e:
                print(f"Process {process_id}: Error processing Query ID {query_id} - {e}")
                error_data = f"Process {process_id} (GPU {cuda_device_id}) ERROR for Query {query_id}: {str(e)}"
                error_bytes = error_data.encode('utf-8')
                error_buffer = ctypes.create_string_buffer(error_bytes)
                error_ptr = ctypes.cast(error_buffer, ctypes.POINTER(ctypes.c_byte))

                output_queue.put({
                    "process_id": process_id,
                    "query_id": query_id,
                    "error": str(e),
                    "cuda_device_attempted": cuda_device_id,
                    "response_ptr": error_ptr, # Even for errors, provide a dummy response
                    "response_size": len(error_bytes)
                })
            finally:
                # Decrement load after processing (even if error)
                worker_status[process_id] = worker_status[process_id] - 1
                print(f"Process {process_id}: Finished Query ID {query_id}. Current load: {worker_status[process_id]}")
            

    except Exception as e:
        print(f"Process {process_id}: Critical error during setup or main loop - {e}")
        output_queue.put({"process_id": process_id, "setup_error": str(e), "cuda_device_attempted": cuda_device_id})


# --- System Under Test (SUT) Class for MLPerf Loadgen ---
class VLLMSchedulingSUT:
    def __init__(self, num_processes: int, num_gpus: int, model_name: str,
                 scheduling_policy: str, max_model_len: int = None, example_data: list = None):
        self.num_processes = num_processes
        self.num_gpus = num_gpus
        self.model_name = model_name
        self.scheduling_policy = scheduling_policy
        self.max_model_len = max_model_len
        self.data = example_data

        # Multiprocessing components
        self.manager = Manager()
        self.worker_input_queues: List[Queue] = []
        self.results_queue = Queue() # Global queue for completed generations
        self.worker_status = self.manager.dict() # Shared dict {worker_id: current_load_in_queue}
        self.processes: List[Process] = []

        self.last_assigned_idx = -1 # For Round Robin scheduling
        self.query_id_to_prompt = {} # To store original prompts by query_id for debugging/tracking

        self._start_workers()
        self._start_result_collector() # Start a thread to continuously collect results

    def _start_workers(self):
        """Starts all vLLM worker processes."""
        print(f"SUT: Starting {self.num_processes} vLLM worker processes...")
        for i in range(self.num_processes):
            worker_id = i + 1
            q = Queue()
            self.worker_input_queues.append(q)
            self.worker_status[worker_id] = 0 # Initialize load for each worker

            assigned_cuda_device_id = i % self.num_gpus
            # Note: 0.9 is for single GPU usage. If multiple processes hit the same GPU,
            # actual utilization will be higher and likely lead to OOM with large models.
            gpu_mem_util = 0.9 / self.num_processes if self.num_gpus == 1 else 0.9 # Crude heuristic

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
                    self.max_model_len
                )
            )
            self.processes.append(process)
            process.start()
            print(f"SUT: Worker {worker_id} started (targets GPU {assigned_cuda_device_id}).")

    def _start_result_collector(self):
        """Starts a separate thread to collect results from workers and report to Loadgen."""
        def collect_and_report():
            while True:
                try:
                    result_data = self.results_queue.get(timeout=0.1) # Wait for results
                    if result_data == "STOP_COLLECTOR":
                        print("SUT Result Collector: Received STOP signal. Exiting.")
                        break

                    query_id = result_data["query_id"]
                    #response_ptr = result_data["response_ptr"]
                    response_size = result_data["response_size"]

                    # Create a Loadgen QuerySampleResponse
                    response = [lg.QuerySampleResponse(query_id, 0, response_size)]
                    lg.QuerySamplesComplete(response) # Inform Loadgen about completion

                    if "error" in result_data or "setup_error" in result_data:
                        error_type = "Setup Error" if "setup_error" in result_data else "Processing Error"
                        error_msg = result_data.get("error") or result_data.get("setup_error")
                        # Corrected line: Use 'or' instead of '||' for Python logical OR
                        print(f"SUT Result Collector: Reported {error_type} for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_attempted') or result_data.get('cuda_device_used')}): {error_msg}")
                    else:
                        print(f"SUT Result Collector: Reported completion for Query {query_id} (Process {result_data.get('process_id')}, GPU {result_data.get('cuda_device_used')}).")

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
        We distribute these queries to our vLLM worker processes based on the scheduling policy.
        """
        print(f"SUT issue_query: Received {len(query_samples)} queries from Loadgen.")
        print(query_samples)
        for q_sample in query_samples:
            query_id = q_sample.id
            print(type(q_sample.id))
            # data_handle is the prompt text in this example
            prompt_text = self.data[q_sample.index]
            print("Query Index",q_sample.id, " Query Index: ",q_sample.index, " -> ", prompt_text);
            self.query_id_to_prompt[q_sample.id] = prompt_text # Store for tracking

            assigned_worker_id = None
            active_worker_ids = [pid for pid in self.worker_status if self.processes[pid-1].is_alive()] # Only consider alive workers

            if not active_worker_ids:
                print(f"SUT issue_query: No active workers available for Query {query_id}. Skipping.")
                # In a real scenario, you'd queue this or handle gracefully.
                # For Loadgen, you must eventually call QuerySamplesComplete for all queries.
                # Here, we'll let it error out or eventually timeout if workers don't come online.
                continue

            if self.scheduling_policy == "first_come_first_served":
                # FCFS: Assign to the first available (idle) worker. If none, wait for one.
                # In offline mode, all queries arrive at once, so this logic needs to be careful.
                # We'll just distribute them in order to the current least loaded (which is essentially FCFS for batch issue)
                # and rely on the least_load logic to pick the first one with 0 load.
                min_load = float('inf')
                least_loaded_worker_id = None
                for wid in active_worker_ids:
                    current_load = self.worker_status[wid]
                    if current_load < min_load:
                        min_load = current_load
                        least_loaded_worker_id = wid
                assigned_worker_id = least_loaded_worker_id

            elif self.scheduling_policy == "least_load":
                min_load = float('inf')
                least_loaded_worker_id = None
                for wid in active_worker_ids:
                    current_load = self.worker_status[wid]
                    if current_load < min_load:
                        min_load = current_load
                        least_loaded_worker_id = wid
                assigned_worker_id = least_loaded_worker_id

            elif self.scheduling_policy == "round_robin":
                self.last_assigned_idx = (self.last_assigned_idx + 1) % len(active_worker_ids)
                assigned_worker_id = active_worker_ids[self.last_assigned_idx]

            
            if assigned_worker_id:
                # Put the query sample data into the selected worker's input queue
                self.worker_input_queues[assigned_worker_id - 1].put({
                    "query_id": query_id,
                    "data_handle": prompt_text # Send the original data handle
                })
                print(f"SUT issue_query: Query ID {query_id} ('{prompt_text[:30]}...') assigned to Process {assigned_worker_id} ({self.scheduling_policy}).")
            else:
                print(f"SUT issue_query: Failed to assign Query ID {query_id}. No suitable worker found.")
            


    def flush_queries(self):
        """
        Callback from Loadgen: Flush any pending queries.
        (Less critical for offline mode, but good practice for other scenarios)
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
        default="least_load",
        choices=["first_come_first_served", "least_load", "round_robin"],
        help="Scheduling policy for distributing prompts to workers in SUT."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples (prompts) Loadgen will issue for the offline test."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/tiny-random-LlamaForCausalLM",
        help="The name of the LLM model to load. Use a tiny model for testing."
    )
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_NAME = args.model_name
    NUM_PROCESSES = args.num_processes
    NUM_GPUS = args.num_gpus
    SCHEDULING_POLICY = args.scheduling_policy
    NUM_SAMPLES = args.num_samples

    if NUM_PROCESSES <= 0:
        print("Error: Number of processes must be at least 1.")
        exit(1)
    if NUM_GPUS <= 0:
        print("Error: Number of GPUs must be at least 1.")
        exit(1)

    # Generate synthetic prompts for Loadgen
    def get_prompt_data(num_samples: int) -> List[bytes]:
        prompts = [f"Simulated prompt {i+1}: What is AI and how does it impact the world?" for i in range(num_samples)]
        #return [p.encode('utf-8') for p in prompts] # Loadgen expects bytes
        return prompts 

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
            example_data = example_prompt_data
        )

        # --- MLPerf Loadgen Setup ---
        # Configure Loadgen loggers

        # Define Loadgen scenario parameters for Offline mode
        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        settings.mode = lg.TestMode.PerformanceOnly # or AccuracyOnly, SubmissionRun
        settings.min_duration_ms = 1000 # Minimum test duration
        settings.min_query_count = NUM_SAMPLES # Ensure all samples are issued

        # Define Loadgen system characteristics (optional, but good practice)

        # Create Loadgen query sample library (QSL)
        # In offline mode, Loadgen gets all data upfront.
        qsl = lg.ConstructQSL(NUM_SAMPLES, NUM_SAMPLES,load_samples_to_ram,unload_samples_from_ram) # Total samples, performance samples
        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries) # Total samples, performance samples
        #qsl.LoadSamplesToRam(example_prompt_data) # Pass our prepared data to QSL

        print(f"MLPerf Loadgen: Starting test with {NUM_SAMPLES} samples in Offline mode...")
        # Note: lg.StartTestWith is not a valid parameter name for lg.StartTest
        # The arguments to lg.StartTest are positional: sut, qsl, settings, system_characteristics
        lg.StartTest(SUTToTest, qsl, settings) # Run the test

        print("\nMLPerf Loadgen test finished.")

    except Exception as e:
        print(f"\nMain program error: {e}")
    finally:
        # --- Clean up ---
        if sut:
            print("Main: Cleaning up SUT resources...")
            sut.stop_workers()
        print("Main: Program finished.")


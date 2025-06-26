import os
import time
import argparse
import logging
import math
from typing import List
from dataset import Dataset
from vllm import TokensPrompt

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


def load_samples_to_ram(query_samples):
    del query_samples
    return

def unload_samples_from_ram(query_samples):
    del query_samples
    return

# --- System Under Test (SUT) Class for MLPerf Loadgen ---
class VLLMSUT:
    def __init__(self, model_name: str, dataset_path: str, max_model_len: int = None):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.max_model_len = max_model_len
        self.data_object = Dataset(self.model_name, dataset_path=self.dataset_path, total_sample_count=24576, device="cpu")
        logging.info("Dataset loaded: %d samples", len(self.data_object.input_ids))
        logging.info("Dataset Max: %d", max(self.data_object.input_lens))
        logging.info("Dataset Min: %d", min(self.data_object.input_lens))
        logging.info("Dataset Len: %d", len(self.data_object.input_lens))
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=4,
            max_model_len=self.max_model_len
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=128,
            stop=["\n\n", "User:", "###", "Human:"]
        )

    def issue_query(self, query_samples: List[lg.QuerySample]):
        logging.info(f"SUT issue_query: Received {len(query_samples)} queries from Loadgen.")
        prompts_to_process = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q_sample.index]) for q_sample in query_samples]
        original_query_ids = [q_sample.id for q_sample in query_samples]
        responses_to_loadgen = []
        try:
            outputs = self.llm.generate(prompts_to_process, self.sampling_params)
            for i, output in enumerate(outputs):
                token_count = len(output.outputs[0].token_ids)
                query_id = original_query_ids[i]
                response = lg.QuerySampleResponse(query_id, 0, 0, token_count)
                responses_to_loadgen.append(response)
        except Exception as e:
            logging.error(f"Error during LLM inference: {e}")
            for query_id in original_query_ids:
                response = lg.QuerySampleResponse(query_id, 0, 0, 0)
                responses_to_loadgen.append(response)
        if responses_to_loadgen:
            lg.QuerySamplesComplete(responses_to_loadgen)
            logging.info(f"SUT: Called QuerySamplesComplete for {len(responses_to_loadgen)} queries.")

    def flush_queries(self):
        logging.info("SUT flush_queries: Flushing (no specific action for offline in this demo).")

# --- Main Program ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run vLLM generation with MLPerf Loadgen in offline scenario.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="(Unused, single process mode)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="(Unused, single process mode)"
    )
    parser.add_argument(
        "--scheduling_policy",
        type=str,
        default="least_load",
        choices=["first_come_first_served", "least_load", "round_robin"],
        help="(Unused, single process mode)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=24576,
        help="Number of samples (prompts) Loadgen will issue for the offline test."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset"
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
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="output-logs", help="Where logs are saved"
    )
    parser.add_argument(
        "--lg_model_name",
        type=str,
        default="llama2-70b",
        choices=["llama2-70b", "llama2-70b-interactive"],
        help="Model name(specified in llm server)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.ERROR),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    MODEL_NAME = args.model_name
    DATASET_PATH = args.dataset_path
    NUM_SAMPLES = args.num_samples
    MAX_MODEL_LEN = args.max_model_len

    if NUM_SAMPLES <= 0:
        logging.error("Error: Number of samples (--num_samples) must be at least 1.")
        exit(1)

    logging.info("-" * 50)

    sut = None
    try:
        sut = VLLMSUT(
            model_name=MODEL_NAME,
            dataset_path=DATASET_PATH,
            max_model_len=MAX_MODEL_LEN
        )

        settings = lg.TestSettings()
        settings.scenario = lg.TestScenario.Offline
        settings.mode = lg.TestMode.PerformanceOnly
        settings.use_token_latencies = True
        logging.info("This may take some time as vLLM model is loaded.")
        settings.FromConfig(args.user_conf, args.lg_model_name, "Offline")
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = args.output_log_dir
        log_output_settings.copy_summary_to_stdout = True
        log_settings = lg.LogSettings()
        log_settings.log_output = log_output_settings
        log_settings.enable_trace = False

        qsl = lg.ConstructQSL(
            24576, # Total samples
            NUM_SAMPLES, # Performance samples
            load_samples_to_ram,
            unload_samples_from_ram
        )

        SUTToTest = lg.ConstructSUT(sut.issue_query, sut.flush_queries)

        logging.info(f"MLPerf Loadgen: Starting test with {NUM_SAMPLES} samples in Offline mode...")
        logging.info(f"Model: {MODEL_NAME}")

        lg.StartTestWithLogSettings(SUTToTest, qsl, settings, log_settings)

        logging.info("\nMLPerf Loadgen test finished.")
        logging.info("Main: Program finished.")
        logging.info("Run Completed!")

    except Exception as e:
        logging.critical(f"\nMain program encountered an error: {e}")


import json
import os
import time
import argparse
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import multiprocessing
from multiprocessing import Manager, Pool, Lock
from functools import partial
import tempfile
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMEmbeddingBenchmark:
    """
    Benchmark for comparing different LLM and embedding models.
    This fills in missing fields in a dataset with actual results from the models.
    """
    
    def __init__(
        self, 
        input_file: str, 
        output_file: str, 
        api_key: str = None,
        max_samples: int = None,
        test_mode: bool = False,
        temperature: float = 0.7,
        top_p: float = 1.0,
        workers: int = 1,
        resume: bool = False
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.max_samples = max_samples
        self.test_mode = test_mode
        self.temperature = temperature
        self.top_p = top_p
        self.workers = workers
        self.resume = resume
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.tokenizer_cache = {}
        
    def get_tokenizer(self, model_name: str):
        """Get the appropriate tokenizer for a model."""
        if model_name not in self.tokenizer_cache:
            try:
                self.tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for newer models not yet in tiktoken
                self.tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        return self.tokenizer_cache[model_name]
    
    def calculate_token_cost(self, text: str, model_name: str) -> float:
        """Calculate the cost of tokens for the given text and model."""
        # Pricing data as of 2024 (update as needed)
        pricing = {
            # Embedding models
            "text-embedding-3-small": 0.00002,  # per 1K tokens
            "text-embedding-3-large": 0.00013,  # per 1K tokens
            # LLM models (GPT-4o models) - approximate pricing
            "gpt-4.1-nano-2025-04-14": {"input": 0.00001, "output": 0.00004},  # per 1K tokens
            "o4-mini-2025-04-16": {"input": 0.00011, "output": 0.00044},  # per 1K tokens
            "gpt-4o-mini-2024-07-18": {"input": 0.000015, "output": 0.00006},  # per 1K tokens
            # Add more models as needed
        }
        
        # Map model aliases to pricing models
        model_mapping = {
            "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano-2025-04-14",
            "o4-mini-2025-04-16": "o4-mini-2025-04-16",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
        }
        
        pricing_model = model_mapping.get(model_name, model_name)
        
        if pricing_model not in pricing:
            logger.warning(f"No pricing data for model {pricing_model}. Using default pricing.")
            if "embedding" in pricing_model:
                return 0.0001 * (len(text) / 4)  # Rough estimate based on chars
            else:
                return 0.001 * (len(text) / 4)  # Rough estimate based on chars
        
        tokenizer = self.get_tokenizer(pricing_model)
        token_count = len(tokenizer.encode(text))
        
        # Calculate cost
        if isinstance(pricing[pricing_model], dict):  # LLM models with input/output pricing
            # For simplicity, assume this is input pricing
            return (pricing[pricing_model]["input"] * token_count) / 1000
        else:  # Embedding models
            return (pricing[pricing_model] * token_count) / 1000
            
    def get_embedding(self, text: str, model: str) -> Tuple[List[float], float, float]:
        """
        Get embeddings for the given text using the specified model.
        Returns: (embedding, latency, cost)
        """
        try:
            # Create a new client for each process
            client = OpenAI(api_key=self.api_key)
            
            # Clean up model name if needed
            if model == "text-embedding-3-small" or model == "text-embedding-3-large":
                model_name = model
            else:
                model_name = "text-embedding-3-small"  # Default fallback
                logger.warning(f"Unknown embedding model {model}, falling back to {model_name}")
            
            start_time = time.time()
            response = client.embeddings.create(
                model=model_name,
                input=text,
                encoding_format="float"
            )
            latency = time.time() - start_time
            embedding = response.data[0].embedding
            cost = self.calculate_token_cost(text, model_name)
            
            return embedding, latency, cost
        except Exception as e:
            logger.error(f"Error getting embedding for model {model}: {e}")
            # Return empty embedding with zero cost in case of error
            return [], 0.0, 0.0
    
    def get_llm_response(self, prompt: str, model_name: str) -> Tuple[str, float, float]:
        """
        Get LLM response for the given prompt using the specified model.
        Returns: (response, latency, cost)
        """
        try:
            # Create a new client for each process
            client = OpenAI(api_key=self.api_key)
            
            start_time = time.time()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p
            )
            latency = time.time() - start_time
            response_text = response.choices[0].message.content
            
            # Calculate costs (input and output)
            input_cost = self.calculate_token_cost(prompt, model_name)
            output_cost = self.calculate_token_cost(response_text, model_name)
            
            # For simplicity, we return total cost
            total_cost = input_cost + output_cost
            
            return response_text, latency, total_cost
        except Exception as e:
            logger.error(f"Error getting LLM response for model {model_name}: {e}")
            # Return empty response with zero cost in case of error
            return "", 0.0, 0.0
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample to fill in missing fields."""
        # Make a copy of the original sample
        result = sample.copy()
        
        prompt = sample.get("Prompt", "")
        embedding_model_a = sample.get("Embedding_Model_A", "")
        embedding_model_b = sample.get("Embedding_Model_B", "")
        llm_a = sample.get("LLM_A", "")
        llm_b = sample.get("LLM_B", "")
        
        # Get embeddings if needed
        if "Prompt_Embedding_A" not in sample or not sample["Prompt_Embedding_A"]:
            embedding_a, latency_a, cost_a = self.get_embedding(prompt, embedding_model_a)
            result["Prompt_Embedding_A"] = embedding_a
            result["Latency_Embedding_Model_A"] = latency_a
            result["Prompt_Embedding_A_Cost"] = cost_a
        
        if "Prompt_Embedding_B" not in sample or not sample["Prompt_Embedding_B"]:
            embedding_b, latency_b, cost_b = self.get_embedding(prompt, embedding_model_b)
            result["Prompt_Embedding_B"] = embedding_b
            result["Latency_Embedding_Model_B"] = latency_b
            result["Prompt_Embedding_B_Cost"] = cost_b
        
        # Get LLM responses if needed
        if "Answer_LLM_A" not in sample or not sample["Answer_LLM_A"]:
            answer_a, latency_a, cost_a = self.get_llm_response(prompt, llm_a)
            result["Answer_LLM_A"] = answer_a
            result["Latency_LLM_A"] = latency_a
            result["Answer_LLM_A_Cost"] = cost_a
        
        if "Answer_LLM_B" not in sample or not sample["Answer_LLM_B"]:
            answer_b, latency_b, cost_b = self.get_llm_response(prompt, llm_b)
            result["Answer_LLM_B"] = answer_b
            result["Latency_LLM_B"] = latency_b
            result["Answer_LLM_B_Cost"] = cost_b
        
        return result
    
    def load_processed_samples(self) -> Set[int]:
        """
        Load already processed samples from the output file to support resuming.
        Returns a set of processed sample IDs.
        """
        processed_ids = set()
        
        if not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0:
            return processed_ids
            
        try:
            with open(self.output_file, 'r') as f:
                for line in f:
                    try:
                        sample = json.loads(line)
                        if "ID" in sample:
                            processed_ids.add(sample["ID"])
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {self.output_file}: {line.strip()}")
                logger.info(f"Found {len(processed_ids)} previously processed samples in {self.output_file}")
        except FileNotFoundError:
            logger.info(f"Output file {self.output_file} not found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading processed samples from {self.output_file}: {e}")
            
        return processed_ids
    
    @staticmethod
    def _save_result_worker(output_file, lock, result):
        """Worker function for saving results by appending to a JSON Lines file."""
        with lock:
            try:
                # Convert result to JSON string
                json_string = json.dumps(result)

                # Append the JSON string as a new line
                with open(output_file, 'a') as f:
                    f.write(json_string + '\n')
                    f.flush()  # Ensure it's written to disk immediately
                    os.fsync(f.fileno()) # Force write to disk
                return True
            except Exception as e:
                logger.error(f"Error saving result to {output_file}: {e}")
                return False
    
    def process_sample_for_mp(self, sample, lock, processed_counter, processed_ids_dict, total_count):
        """Process a single sample and save the result, designed for multiprocessing."""
        try:
            # Skip if already processed and resuming
            sample_id = sample.get("ID")
            if self.resume and sample_id is not None and sample_id in processed_ids_dict:
                with lock:
                    processed_counter.value += 1
                return sample
            
            # Process the sample
            result = self.process_sample(sample)
            
            # Save the result
            success = self._save_result_worker(self.output_file, lock, result)
            
            # Update the processed counter
            with lock:
                processed_counter.value += 1
                if "ID" in result:
                    processed_ids_dict[result["ID"]] = True
            
            return result
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            with lock:
                processed_counter.value += 1
            # Return the original sample if processing fails
            return sample
    
    def _update_progress_bar(self, pbar, processed_counter, stop_event):
        """Thread function to update progress bar based on shared counter value."""
        last_value = 0
        while not stop_event.is_set():
            current_value = processed_counter.value
            if current_value > last_value:
                pbar.update(current_value - last_value)
                last_value = current_value
            time.sleep(0.1)  # Small sleep to avoid CPU hogging
    
    def _convert_json_to_jsonl(self, file_path: str):
        """Converts a file containing a JSON array to JSON Lines format."""
        logger.info(f"Converting existing JSON file {file_path} to JSON Lines format...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning(f"File {file_path} is not a JSON array. Skipping conversion.")
                return

            # Write to a temporary file first
            temp_dir = os.path.dirname(file_path)
            with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False, suffix=".jsonl") as temp_f:
                temp_path = temp_f.name
                for item in data:
                    temp_f.write(json.dumps(item) + '\n')
                temp_f.flush()
                os.fsync(temp_f.fileno())

            # Replace original file with the converted temp file
            os.replace(temp_path, file_path)
            logger.info(f"Successfully converted {file_path} to JSON Lines format.")

        except json.JSONDecodeError:
            logger.warning(f"File {file_path} is not valid JSON. Assuming it might already be JSON Lines or corrupted. Skipping conversion.")
        except Exception as e:
            logger.error(f"Error converting {file_path} to JSON Lines: {e}")
            # Attempt to clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise  # Re-raise the exception

    def run_benchmark(self):
        """Run the benchmark to fill in missing fields in the dataset."""
        try:
            # Load input data
            logger.info(f"Loading input data from {self.input_file}")
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            
            # Test mode - only process the first sample
            if self.test_mode:
                logger.info("Running in test mode - only processing the first sample")
                data = data[:1]
            # Limit samples if specified
            elif self.max_samples and self.max_samples < len(data):
                logger.info(f"Limiting to {self.max_samples} samples out of {len(data)}")
                data = data[:self.max_samples]
            
            total_count = len(data)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
            
            # Output file is appended to, so no initial '[]' needed.
            # Just ensure the file exists if we are not resuming or it's empty.
            if not self.resume and (not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0):
                 # Create the file if it doesn't exist to avoid FileNotFoundError later
                 open(self.output_file, 'a').close()
            
            # Set up multiprocessing shared objects
            manager = Manager()
            processed_ids_dict = manager.dict()
            lock = manager.Lock()
            processed_counter = manager.Value('i', 0)
            
            # Load already processed samples for resuming
            if self.resume:
                # Check if output file exists and needs conversion from JSON array to JSON Lines
                if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                    try:
                        with open(self.output_file, 'r') as f:
                            # Check the first non-whitespace character
                            first_char = ""
                            for line in f:
                                stripped_line = line.strip()
                                if stripped_line:
                                    first_char = stripped_line[0]
                                    break
                        if first_char == '[':
                            self._convert_json_to_jsonl(self.output_file)
                    except Exception as e:
                         logger.error(f"Error checking or converting output file {self.output_file}: {e}")
                         # Decide how to proceed - perhaps halt or try to continue assuming JSONL
                         logger.warning("Attempting to proceed assuming JSON Lines format despite error during check/conversion.")

                for id in self.load_processed_samples():
                    processed_ids_dict[id] = True
                pending_count = len(data) - len(processed_ids_dict)
                if pending_count <= 0:
                    logger.info("All samples have already been processed. Nothing to do.")
                    return
                logger.info(f"Resuming processing. {len(processed_ids_dict)} samples already processed, {pending_count} remaining.")
            
            # Set up progress bar in the main thread
            pbar = tqdm(total=total_count, desc="Processing samples")
            stop_event = threading.Event()
            
            # Create a thread to update the progress bar
            progress_thread = threading.Thread(
                target=self._update_progress_bar, 
                args=(pbar, processed_counter, stop_event)
            )
            progress_thread.daemon = True
            progress_thread.start()
            
            try:
                # Process samples
                num_workers = min(self.workers, len(data))
                if num_workers > 1:
                    logger.info(f"Processing {len(data)} samples with {num_workers} parallel workers (temperature={self.temperature}, top_p={self.top_p})")
                    
                    # Use a process pool to process samples in parallel
                    with Pool(processes=num_workers) as pool:
                        process_func = partial(
                            self.process_sample_for_mp,
                            lock=lock,
                            processed_counter=processed_counter,
                            processed_ids_dict=processed_ids_dict,
                            total_count=total_count
                        )
                        results = pool.map(process_func, data)
                else:
                    logger.info(f"Processing {len(data)} samples sequentially (temperature={self.temperature}, top_p={self.top_p})")
                    
                    # Process samples sequentially
                    results = []
                    for sample in data:
                        result = self.process_sample_for_mp(
                            sample, 
                            lock,
                            processed_counter,
                            processed_ids_dict,
                            total_count
                        )
                        results.append(result)
            finally:
                # Stop the progress bar thread
                stop_event.set()
                progress_thread.join(timeout=1.0)
                pbar.close()
            
            # Report completion stats
            if self.resume:
                logger.info(f"Benchmark completed. Processed {processed_counter.value - len(processed_ids_dict)} new samples. Results saved to {self.output_file}")
            else:
                logger.info(f"Benchmark completed. Results saved to {self.output_file}")
                
            return results
        
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM and Embedding Comparison Benchmark")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input JSON file containing partial samples")
    parser.add_argument("--output", type=str, 
                        help="Path to output JSON file (default: input_filled_TIMESTAMP.jsonl)")
    parser.add_argument("--api-key", type=str, 
                        help="OpenAI API key (if not set as environment variable)")
    parser.add_argument("--max-samples", type=int, 
                        help="Maximum number of samples to process")
    parser.add_argument("--test", action="store_true",
                        help="Test mode - only process the first sample")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for LLM response generation (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p (nucleus sampling) value for LLM response generation (default: 1.0)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for processing samples (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume processing from where it was stopped")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set default output file if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_basename = os.path.basename(args.input)
        input_name = os.path.splitext(input_basename)[0]
        # Default to .jsonl extension
        args.output = f"{input_name}_filled_{timestamp}.jsonl"
    
    # Create benchmark and run
    benchmark = LLMEmbeddingBenchmark(
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        max_samples=args.max_samples,
        test_mode=args.test,
        temperature=args.temperature,
        top_p=args.top_p,
        workers=args.workers,
        resume=args.resume
    )
    
    benchmark.run_benchmark()

if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main() 
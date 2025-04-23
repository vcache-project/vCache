import json
import os
import time
import argparse
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import tiktoken
import concurrent.futures
from functools import partial
import threading
import tempfile

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
        self.file_lock = threading.Lock()
        self.processed_count = 0
        self.total_count = 0
        self.processed_ids = set()  # Track IDs of processed samples
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass as parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
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
            
    async def get_embedding(self, text: str, model: str) -> Tuple[List[float], float, float]:
        """
        Get embeddings for the given text using the specified model.
        Returns: (embedding, latency, cost)
        """
        try:
            # Clean up model name if needed
            if model == "text-embedding-3-small" or model == "text-embedding-3-large":
                model_name = model
            else:
                model_name = "text-embedding-3-small"  # Default fallback
                logger.warning(f"Unknown embedding model {model}, falling back to {model_name}")
            start_time = time.time()
            response = self.client.embeddings.create(
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
    
    async def get_llm_response(self, prompt: str, model_name: str) -> Tuple[str, float, float]:
        """
        Get LLM response for the given prompt using the specified model.
        Returns: (response, latency, cost)
        """
        try:
            start_time = time.time()
            response = self.client.chat.completions.create(
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
    
    async def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
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
            embedding_a, latency_a, cost_a = await self.get_embedding(prompt, embedding_model_a)
            result["Prompt_Embedding_A"] = embedding_a
            result["Latency_Embedding_Model_A"] = latency_a
            result["Prompt_Embedding_A_Cost"] = cost_a
        
        if "Prompt_Embedding_B" not in sample or not sample["Prompt_Embedding_B"]:
            embedding_b, latency_b, cost_b = await self.get_embedding(prompt, embedding_model_b)
            result["Prompt_Embedding_B"] = embedding_b
            result["Latency_Embedding_Model_B"] = latency_b
            result["Prompt_Embedding_B_Cost"] = cost_b
        
        # Get LLM responses if needed
        if "Answer_LLM_A" not in sample or not sample["Answer_LLM_A"]:
            answer_a, latency_a, cost_a = await self.get_llm_response(prompt, llm_a)
            result["Answer_LLM_A"] = answer_a
            result["Latency_LLM_A"] = latency_a
            result["Answer_LLM_A_Cost"] = cost_a
        
        if "Answer_LLM_B" not in sample or not sample["Answer_LLM_B"]:
            answer_b, latency_b, cost_b = await self.get_llm_response(prompt, llm_b)
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
                try:
                    processed_samples = json.load(f)
                    for sample in processed_samples:
                        if "ID" in sample:
                            processed_ids.add(sample["ID"])
                    logger.info(f"Found {len(processed_ids)} previously processed samples")
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {self.output_file}. Starting from scratch.")
        except Exception as e:
            logger.error(f"Error loading processed samples: {e}")
            
        return processed_ids
    
    def save_result(self, result: Dict[str, Any]):
        """
        Save a single result to the output file.
        Uses a file lock to ensure thread safety.
        """
        with self.file_lock:
            self.processed_count += 1
            
            # Create a temp file to avoid corruption if the process is interrupted while writing
            temp_dir = os.path.dirname(self.output_file)
            with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as temp_file:
                temp_path = temp_file.name
                
                try:
                    if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                        # File exists and has content, read existing results
                        with open(self.output_file, 'r') as f:
                            try:
                                results = json.load(f)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON from {self.output_file}. Creating new results array.")
                                results = []
                    else:
                        # File doesn't exist or is empty
                        results = []
                    
                    # Append the new result
                    results.append(result)
                    
                    # Add the ID to processed IDs set
                    if "ID" in result:
                        self.processed_ids.add(result["ID"])
                    
                    # Write to temp file
                    json.dump(results, temp_file, indent=2)
                    
                    # Ensure data is written to disk
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                except Exception as e:
                    logger.error(f"Error saving result: {e}")
                    return False
            
            # Atomically replace the output file with the temp file
            try:
                os.replace(temp_path, self.output_file)
                logger.debug(f"Saved result {self.processed_count}/{self.total_count} to {self.output_file}")
                return True
            except Exception as e:
                logger.error(f"Error replacing output file: {e}")
                # Try to clean up the temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                return False
    
    async def process_sample_wrapper(self, sample: Dict[str, Any], pbar: tqdm = None) -> Dict[str, Any]:
        """Wrapper for process_sample to update progress bar, save result, and handle errors."""
        try:
            # Skip if already processed and resuming
            if self.resume and "ID" in sample and sample["ID"] in self.processed_ids:
                if pbar:
                    pbar.update(1)
                return sample
            
            result = await self.process_sample(sample)
            
            # Save the result immediately
            self.save_result(result)
            
            if pbar:
                pbar.update(1)
            return result
        except Exception as e:
            logger.error(f"Error processing sample: {e}")
            if pbar:
                pbar.update(1)
            # Return the original sample if processing fails
            return sample
    
    async def run_benchmark(self):
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
            
            self.total_count = len(data)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.output_file)), exist_ok=True)
            
            # Initialize output file with an empty JSON array if it doesn't exist
            if not os.path.exists(self.output_file):
                with open(self.output_file, 'w') as f:
                    json.dump([], f)
            
            # Load already processed samples for resuming
            if self.resume:
                self.processed_ids = self.load_processed_samples()
                pending_count = len(data) - len(self.processed_ids)
                if pending_count <= 0:
                    logger.info("All samples have already been processed. Nothing to do.")
                    return
                logger.info(f"Resuming processing. {len(self.processed_ids)} samples already processed, {pending_count} remaining.")
            
            # Process samples
            num_workers = min(self.workers, len(data))
            if num_workers > 1:
                logger.info(f"Processing {len(data)} samples with {num_workers} parallel workers (temperature={self.temperature}, top_p={self.top_p})")
            else:
                logger.info(f"Processing {len(data)} samples sequentially (temperature={self.temperature}, top_p={self.top_p})")
            
            # Initialize progress bar
            pbar = tqdm(total=len(data), desc="Processing samples")
            
            # Process samples in parallel
            results = []
            if num_workers > 1:
                # Use a thread pool to process samples concurrently
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Create tasks
                    futures = [
                        loop.run_in_executor(
                            executor,
                            partial(
                                asyncio.run,
                                self.process_sample_wrapper(sample, pbar)
                            )
                        )
                        for sample in data
                    ]
                    
                    # Wait for all tasks to complete
                    completed_results = await asyncio.gather(*futures)
                    results.extend(completed_results)
            else:
                # Process samples sequentially
                for sample in data:
                    result = await self.process_sample_wrapper(sample, pbar)
                    results.append(result)
            
            pbar.close()
            
            # Report completion stats
            if self.resume:
                logger.info(f"Benchmark completed. Processed {self.processed_count} new samples. Results saved to {self.output_file}")
            else:
                logger.info(f"Benchmark completed. Results saved to {self.output_file}")
        
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM and Embedding Comparison Benchmark")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to input JSON file containing partial samples")
    parser.add_argument("--output", type=str, 
                        help="Path to output JSON file (default: input_filled_TIMESTAMP.json)")
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

async def main():
    args = parse_arguments()
    
    # Set default output file if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_basename = os.path.basename(args.input)
        input_name = os.path.splitext(input_basename)[0]
        args.output = f"{input_name}_filled_{timestamp}.json"
    
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
    
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main()) 
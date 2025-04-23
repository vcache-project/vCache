import json
import os
import time
import argparse
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import tiktoken

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
        max_samples: int = None
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.max_samples = max_samples
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
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
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
            "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
            # Add more models as needed
        }
        
        # Map model aliases to pricing models
        model_mapping = {
            "4o-mini": "gpt-4o-mini",
            "4o": "gpt-4o",
            # Add more mappings as needed
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
        start_time = time.time()
        try:
            # Clean up model name if needed
            if model == "text-embedding-3-small" or model == "text-embedding-3-large":
                model_name = model
            else:
                model_name = "text-embedding-3-small"  # Default fallback
                logger.warning(f"Unknown embedding model {model}, falling back to {model_name}")
            
            response = self.client.embeddings.create(
                model=model_name,
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            latency = time.time() - start_time
            cost = self.calculate_token_cost(text, model_name)
            
            return embedding, latency, cost
        except Exception as e:
            logger.error(f"Error getting embedding for model {model}: {e}")
            # Return empty embedding with zero cost in case of error
            return [], 0.0, 0.0
    
    async def get_llm_response(self, prompt: str, model: str) -> Tuple[str, float, float]:
        """
        Get LLM response for the given prompt using the specified model.
        Returns: (response, latency, cost)
        """
        start_time = time.time()
        try:
            # Clean up model name if needed
            if model == "4o-mini":
                model_name = "gpt-4o-mini"
            elif model == "4o":
                model_name = "gpt-4o"
            else:
                model_name = "gpt-4o-mini"  # Default fallback
                logger.warning(f"Unknown LLM model {model}, falling back to {model_name}")
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            latency = time.time() - start_time
            
            # Calculate costs (input and output)
            input_cost = self.calculate_token_cost(prompt, model_name)
            output_cost = self.calculate_token_cost(response_text, model_name)
            
            # For simplicity, we return total cost
            total_cost = input_cost + output_cost
            
            return response_text, latency, total_cost
        except Exception as e:
            logger.error(f"Error getting LLM response for model {model}: {e}")
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
    
    async def run_benchmark(self):
        """Run the benchmark to fill in missing fields in the dataset."""
        try:
            # Load input data
            logger.info(f"Loading input data from {self.input_file}")
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            
            # Limit samples if specified
            if self.max_samples and self.max_samples < len(data):
                logger.info(f"Limiting to {self.max_samples} samples out of {len(data)}")
                data = data[:self.max_samples]
            
            # Process samples
            logger.info(f"Processing {len(data)} samples")
            results = []
            
            for sample in tqdm(data, desc="Processing samples"):
                filled_sample = await self.process_sample(sample)
                results.append(filled_sample)
                
                # Optional: add delay to avoid API rate limits
                await asyncio.sleep(0.1)
            
            # Save results
            logger.info(f"Saving results to {self.output_file}")
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
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
        max_samples=args.max_samples
    )
    
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main()) 
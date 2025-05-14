import json
import sys
import os

def transform_json_object(original_obj):
    # Mappings: Original Key (from "Mapped column") -> New Key (from "Target field")
    mapping_from_original = {
        "ID": "id",
        "Prompt": "text",
        "Answer_LLM_A": "response_1",
        "Prompt_Embedding_A": "embedding_1",
        "Answer_LLM_B": "response_2",
        "Prompt_Embedding_B": "embedding_2",
        "Latency_LLM_A": "response_1_lat",
        "Latency_Embedding_Model_A": "embedding_1_lat",
        "Latency_LLM_B": "response_2_lat",
        "Latency_Embedding_Model_B": "embedding_2_lat"
    }

    transformed_obj = {}

    # Apply mappings from the original object
    for original_key, target_key in mapping_from_original.items():
        if original_key in original_obj:
            transformed_obj[target_key] = original_obj[original_key]
    
    # Add new field "dataset_name" with value "lm arena"
    transformed_obj["dataset_name"] = "lm arena"
    
    # Set "output_format" to an empty string
    transformed_obj["output_format"] = ""
    
    return transformed_obj

def main():
    input_file_path = '/data/acuadron/VectorQ/benchmarks/data/benchmark_fill_shuffled.jsonl'
    output_file_path = '/data/acuadron/VectorQ/benchmarks/data/benchmark_fill_transformed.jsonl'
    abs_output_path = os.path.abspath(output_file_path)

    print(f"Starting transformation from '{input_file_path}' to '{output_file_path}' (absolute: '{abs_output_path}')...")
    lines_processed = 0
    lines_skipped_decode_error = 0
    lines_skipped_processing_error = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(infile):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    continue
                try:
                    original_data = json.loads(stripped_line)
                    transformed_data = transform_json_object(original_data)
                    outfile.write(json.dumps(transformed_data) + '\\n')
                    lines_processed += 1
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"Skipping line {line_number + 1} due to JSON decode error: {e}. Line content: '{stripped_line[:100]}...'\\n")
                    lines_skipped_decode_error += 1
                except Exception as e:
                    sys.stderr.write(f"Error processing line {line_number + 1}: {e}. Line content: '{stripped_line[:100]}...'\\n")
                    lines_skipped_processing_error += 1
                
                if (line_number + 1) % 1000 == 0: # Print progress every 1000 lines
                    print(f"Processed {line_number + 1} lines...")

    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file '{input_file_path}' not found.\\n")
        sys.exit(1)
    except IOError as e:
        sys.stderr.write(f"IOError: {e}\\n")
        sys.exit(1)

    print(f"Processing complete.")
    print(f"Total lines processed successfully: {lines_processed}")
    print(f"Total lines skipped (JSON decode error): {lines_skipped_decode_error}")
    print(f"Total lines skipped (other processing error): {lines_skipped_processing_error}")
    print(f"Transformed data written to '{abs_output_path}'")

if __name__ == "__main__":
    main() 
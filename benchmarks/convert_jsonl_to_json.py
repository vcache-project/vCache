import json
import os
import sys # Added for stderr

# Copied and adapted from transform_jsonl.py
def transform_json_object(original_obj):
    # Mappings: Original Key (from "Mapped column") -> New Key (from "Target field")
    mapping_from_original = {
        "ID": "id",
        "Prompt": "task", # This was "Prompt" in the original, assuming "text" is the target
        "Answer_LLM_A": "response_1",
        "Prompt_Embedding_A": "embedding_1",
        "Answer_LLM_B": "response_2",
        "Prompt_Embedding_B": "embedding_2",
        "Latency_LLM_A": "response_1_lat",
        "Latency_Embedding_Model_A": "embedding_1_lat",
        "Latency_LLM_B": "response_2_lat",
        "Latency_Embedding_Model_B": "embedding_2_lat",
        "ID_Set": "ID_Set",
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
    # Ensure "task" is present, defaulting to empty string if not in original for some reason
    # Based on benchmark.py, "task" might come from the original data, 
    # but if not, it's good practice for it to exist if other parts of the system expect it.
    # If it's always in the input from benchmark_fill_transformed.jsonl, this line is just a safe default.
    transformed_obj["text"] = original_obj.get("Prompt", "") 
    
    return transformed_obj

def convert_jsonl_to_json_with_transformation(input_jsonl_path, output_json_path):
    """
    Converts a JSONL file to a standard JSON array file, applying transformations.

    Args:
        input_jsonl_path (str): Path to the input JSONL file.
        output_json_path (str): Path to the output JSON file.
    """
    print(f"Starting conversion and transformation of {input_jsonl_path} to {output_json_path}...")
    lines_processed_successfully = 0
    lines_skipped_decode_error = 0
    lines_skipped_transform_error = 0

    try:
        jsons = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile, \
             open(output_json_path, 'w', encoding='utf-8') as outfile:

            print("Output file opened, writing initial '['.")

            first_line = True
            for i, line_content in enumerate(infile):
                stripped_line = line_content.strip()
                if not stripped_line: # Skip empty lines
                    continue
                
                try:
                    original_data = json.loads(stripped_line)

                    transformed_data = transform_json_object(original_data)
                    jsons.append(transformed_data)

                    if (i + 1) % 10000 == 0: # Print progress every 10,000 lines read
                        print(f"Attempted to process {i + 1} lines...")
                    
                    lines_processed_successfully += 1
                    first_line = False

                except json.JSONDecodeError as e:
                    sys.stderr.write(f"Skipping line {i+1} due to JSON decode error: {e}. Line: '{stripped_line[:100]}...'\n")
                    lines_skipped_decode_error += 1
                    # If a line is skipped, we need to ensure we don't have a leading comma issue later.
                    # The current logic handles comma *before* writing, so a skipped line won't cause an extra comma.
                except Exception as e:
                    exit(0)
                    sys.stderr.write(f"Skipping line {i+1} due to transformation error: {e}. Data: '{str(original_data)[:100]}...'\n")
                    lines_skipped_transform_error +=1

                if (i + 1) % 10000 == 0: # Print progress every 10,000 lines read
                    print(f"Attempted to process {i + 1} lines...")
            
            
            outfile.write(json.dumps(jsons))
            print("Finished writing all lines, adding closing ']' and closing output file.")

        print(f"Successfully converted and transformed {input_jsonl_path} to {output_json_path}")
        print(f"Total lines processed successfully: {lines_processed_successfully}")
        print(f"Total lines skipped (JSON decode error): {lines_skipped_decode_error}")
        print(f"Total lines skipped (transformation error): {lines_skipped_transform_error}")


    except FileNotFoundError:
        print(f"Error: Input file not found at {input_jsonl_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Determine the base directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full paths relative to the script's location
    # Assumes 'data/large_scale/' is a subdirectory relative to where the script is.
    # If the script is in 'benchmarks/', and data is in 'benchmarks/data/large_scale'
    
    # Path to the /data/acuadron/VectorQ/benchmarks/data/large_scale/ directory
    # This should be an absolute path or correctly relative from where the script is run.
    # Given the user context, an absolute path is safer here.
    base_data_dir = "/data/acuadron/VectorQ/benchmarks/data/large_scale/"
    
    input_file = os.path.join(base_data_dir, "benchmark_fill_shuffled.jsonl")
    output_file = os.path.join(base_data_dir, "benchmark_fill_final.json")

    convert_jsonl_to_json_with_transformation(input_file, output_file) 
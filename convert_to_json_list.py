import json
import sys

def convert_jsonl_to_json_list(input_file_path, output_file_path):
    print(f"Starting conversion from JSONL '{input_file_path}' to JSON list '{output_file_path}'...")
    json_list = []
    lines_processed = 0
    lines_skipped_decode_error = 0

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(infile):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    continue
                try:
                    json_obj = json.loads(stripped_line)
                    json_list.append(json_obj)
                    lines_processed += 1
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"Skipping line {line_number + 1} due to JSON decode error: {e}. Line content: '{stripped_line[:100]}...'\\n")
                    lines_skipped_decode_error += 1
                
                if (line_number + 1) % 10000 == 0: # Print progress every 10,000 lines
                    print(f"Read {line_number + 1} lines from input...")

        print(f"Finished reading input file. Total JSON objects collected: {len(json_list)}")
        print(f"Now writing to output file '{output_file_path}'...")
        
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(json_list, outfile, indent=2) # Using indent=2 for readability

        print(f"Conversion complete.")
        print(f"Total JSON objects written: {len(json_list)}")
        print(f"Total lines skipped (JSON decode error): {lines_skipped_decode_error}")
        print(f"Output written to '{output_file_path}'")

    except FileNotFoundError:
        sys.stderr.write(f"Error: Input file '{input_file_path}' not found.\\n")
        sys.exit(1)
    except IOError as e:
        sys.stderr.write(f"IOError while reading '{input_file_path}' or writing to '{output_file_path}': {e}\\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"An unexpected error occurred: {e}\\n")
        sys.exit(1)

def main():
    input_jsonl_path = '/data/acuadron/VectorQ/benchmarks/data/benchmark_fill_transformed.jsonl'
    output_json_path = '/data/acuadron/VectorQ/benchmarks/data/benchmark_fill_final.json'
    convert_jsonl_to_json_list(input_jsonl_path, output_json_path)

if __name__ == "__main__":
    main() 
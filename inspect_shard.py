import gzip
import json

def inspect_and_save_sample(shard_path, sample_path, num_lines=10):
    with gzip.open(shard_path, 'rt', encoding='utf-8') as in_file:
        with gzip.open(sample_path, 'wt', encoding='utf-8') as out_file:
            for i in range(num_lines):
                line = in_file.readline()
                if not line:
                    break  # End of file
                try:
                    json_document = json.loads(line)
                    print(json.dumps(json_document, indent=2, ensure_ascii=False))  # Pretty print JSON
                    out_file.write(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    continue

if __name__ == "__main__":
    shard_path = input('Enter the path to the shard file (e.g., uk-shards/uk_all_resharded.json.gz): ')
    sample_path = input('Enter the path to save the sample file (e.g., uk-shards/shard_sample.json.gz): ')
    num_lines = int(input('Enter the number of lines to inspect and save (default is 10): ') or 10)
    inspect_and_save_sample(shard_path, sample_path, num_lines)
    print(f'Sample saved to {sample_path}')
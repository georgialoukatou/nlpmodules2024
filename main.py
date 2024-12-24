import json

input_file = 'reviews.jsonl'  # Replace with your input file name
output_file = 'filtered_reviews.jsonl'  # Output file for filtered reviews
target_asin = 'B07Y1B4NH6'#'B09HFW8GRB'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'a', encoding='utf-8') as outfile:
    for line in infile:
        try:
            review = json.loads(line)
            if review.get('parent_asin') == target_asin:
                print("found")
                json.dump(review, outfile)
                outfile.write('\n')
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")

input_file = 'meta.jsonl'  # Replace with your input file name
output_file = 'filtered_meta.jsonl'  # Output file for filtered reviews


with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'a', encoding='utf-8') as outfile:
    for line in infile:
        try:
            review = json.loads(line)
            if review.get('parent_asin') == target_asin:
                print("found")
                json.dump(review, outfile)
                outfile.write('\n')
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")

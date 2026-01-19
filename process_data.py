import json
import os

def process_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'chat_safe_clean.json')
    output_path = os.path.join(base_dir, 'data', 'train.json')
    
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return

    training_pairs = []
    
    # Iterate through the list and find User -> Assistant pairs
    i = 0
    skipped_count = 0
    while i < len(data) - 1:
        current_msg = data[i]
        next_msg = data[i+1]
        
        if current_msg['role'] == 'user' and next_msg['role'] == 'assistant':
            # Found a valid pair
            instruction = current_msg['content']
            # Optional: Clean up instruction if needed
            
            output = next_msg['content']
            # Optional: Clean up output
            
            training_pairs.append({
                "instruction": instruction,
                "input": "", # No specific context input for now
                "output": output
            })
            i += 2
        else:
            # If not a pair (e.g. User -> User, or Assistant -> User which implies we started on an Assistant msg)
            # If current is Assistant, just skip it to find next User.
            # If current is User but next is User, skip current User (or combine?)
            # For simplicity, just skip to next index
            # print(f"Skipping non-pair at index {i}: {current_msg['role']} -> {next_msg['role']}")
            skipped_count += 1
            i += 1

    print(f"✅ Processed {len(data)} unique messages.")
    print(f"✅ Created {len(training_pairs)} training pairs.")
    print(f"⚠️ Skipped {skipped_count} unpaired messages.")
    
    # Save to train.json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, ensure_ascii=False, indent=4)
        
    print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    process_data()

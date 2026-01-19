import json
import os
import subprocess
import time
import sys

def prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'data', 'train.json')
    output_path = os.path.join(base_dir, 'data', 'train.txt')
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Converting {len(data)} items to Llama 3 text format...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # Llama 3 Instruct Format
            # <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>
            
            instruction = item.get('instruction', '')
            user_input = item.get('input', '')
            output = item.get('output', '')
            
            full_prompt = instruction
            if user_input:
                full_prompt += "\n" + user_input
                
            text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{full_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>\n"
            )
            f.write(text)
            
    print(f"Saved training text to {output_path}")
    return output_path

def run_training():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'data', 'train.txt')
    model_path = "e:/AI/models/Llama-3-8B-Instruct-Q4_K_M.gguf"
    output_model = os.path.join(base_dir, 'outputs', 'My-L3-Finetune.gguf')
    
    # Path to llama-finetune.exe
    # Assuming standard build path from workspace info
    exe_path = "e:/AI/llama.cpp/build/bin/Release/llama-finetune.exe"
    
    if not os.path.exists(exe_path):
        print(f"❌ Error: Finetune binary not found at {exe_path}")
        return

    # Args for finetune
    # Note: finetune.exe does full finetuning.
    # It might create a checkpoint or a new model file.
    # We use --epochs 1 to keep it short (approx coverage of data).
    # Since we can't specify steps easily, we rely on epochs.
    # 282 items allows for quick 1 epoch.
    
    # Arg format based on help:
    # llama-finetune -m model -f data -o output ...
    
    cmd = [
        exe_path,
        "--model", model_path,
        "--file", train_file,
        "--output-file", output_model,
        "--ctx-size", "2048",   # Context size
        "--threads", "6",       # Threads
        "--epochs", "1",        # Local epochs
        "--adam-iter", "100",   # Wait, help didn't show this, but maybe it works? 
                                # If not, we rely on epochs.
                                # Actually, standard finetune usually takes --adam-iter N for steps. 
                                # Let's try passing it. If it fails, we remove it.
        "--use-checkpointing",  # Gradient checkpointing to save RAM? (Check help? Not listed).
        "--batch-size", "1"
    ]
    
    # Filter args that might not exist based on help output:
    # Help showed: -o, -lr, -epochs, -opt.
    # Doesn't explicitly show --adam-iter.
    # But let's try just passing -epochs 1.
    
    cmd = [
        exe_path,
        "--model", model_path,
        "--file", train_file,
        "--output-file", output_model,
        "--ctx-size", "1024",
        "--threads", "6",
        "--epochs", "1", 
        "--val-split", "0.05"
    ]

    print(f"🚀 Starting Training (Reference: 100 steps -> 1 Epoch approx)...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        step_counter = 0
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                # Try to detect progress (some versions print "iter X")
                if "iter" in line or "loss" in line:
                    step_counter += 1
                    # Note: Depending on verbosity, it might print often.
                    
        rc = process.poll()
        if rc == 0:
            print("\n✅ Training Completed Successfully!")
        else:
            print(f"\n❌ Training Failed with exit code {rc}")
            
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    prepare_data()
    run_training()

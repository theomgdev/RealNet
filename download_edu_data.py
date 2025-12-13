import os
import requests
from datasets import load_dataset
import tqdm

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_gsm8k(out_path):
    print("Downloading GSM8K (Math)...")
    try:
        # Load from Hugging Face
        dataset = load_dataset("openai/gsm8k", "main")
        
        with open(out_path, 'w', encoding='utf-8') as f:
            # Combine train and test for "General Knowledge"
            for split in ['train', 'test']:
                for item in dataset[split]:
                    question = item['question']
                    answer = item['answer']
                    # Format: Q: ... \n A: ...
                    f.write(f"Question: {question}\nAnswer: {answer}\n\n")
        print(f"GSM8K saved to {out_path}")
    except Exception as e:
        print(f"Failed to download GSM8K: {e}")

def download_tinystories(out_path):
    print("Downloading TinyStories (Language)...")
    try:
        # TinyStories is large, sticking to a subset or streaming
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        
        limit = 50000 # Start with 50k stories (approx 10-20MB?)
        count = 0
        
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                text = item['text']
                f.write(text + "\n\n<|endofstory|>\n\n")
                count += 1
                if count >= limit:
                    break
                    
        print(f"TinyStories (First {limit}) saved to {out_path}")
    except Exception as e:
        print(f"Failed to download TinyStories: {e}")

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    ensure_dir(data_dir)
    
    gsm8k_path = os.path.join(data_dir, 'gsm8k.txt')
    tiny_path = os.path.join(data_dir, 'tinystories.txt')
    
    # 1. Math
    if not os.path.exists(gsm8k_path):
        download_gsm8k(gsm8k_path)
    else:
        print("GSM8K already exists.")
        
    # 2. Language
    if not os.path.exists(tiny_path):
        download_tinystories(tiny_path)
    else:
        print("TinyStories already exists.")
        
    print("Educational Data Ready! 🎓")

if __name__ == '__main__':
    main()

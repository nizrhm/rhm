import os
import json
import torch
import zipfile
import urllib.request
from torch.utils.data import Dataset

# --- ARC VOCABULARY ---
# Tokens 0-9 are reserved for the 10 ARC colors.
PAD_TOKEN = 10
ROW_TOKEN = 11
INPUT_TOKEN = 12
OUTPUT_TOKEN = 13
TEST_TOKEN = 14

class ARCDataset(Dataset):
    def __init__(self, data_dir="./arc_data/training", max_seq_len=2048):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.tasks = []
        
        # 1. Download data if it doesn't exist
        self._ensure_data_downloaded()
        
        # 2. Load all JSON files
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    self.tasks.append(json.load(f))
                    
        print(f"Successfully loaded {len(self.tasks)} ARC training tasks.")

    def _ensure_data_downloaded(self):
        """Downloads the official ARC dataset from GitHub if not present locally."""
        if not os.path.exists(self.data_dir):
            print("Downloading official ARC-AGI dataset...")
            url = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"
            zip_path = "arc_master.zip"
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("./")
            
            # Rename for cleaner pathing
            os.rename("ARC-AGI-master/data", "./arc_data")
            os.remove(zip_path)
            print("Download and extraction complete.")

    def _flatten_grid(self, grid):
        """Converts a 2D grid into a 1D list with ROW tokens at the end of each row."""
        flat = []
        for row in grid:
            flat.extend(row)
            flat.append(ROW_TOKEN)
        return flat

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        sequence = []
        
        # 1. Add Context Demonstrations (Input -> Output pairs)
        for demo in task['train']:
            sequence.append(INPUT_TOKEN)
            sequence.extend(self._flatten_grid(demo['input']))
            
            sequence.append(OUTPUT_TOKEN)
            sequence.extend(self._flatten_grid(demo['output']))
            
        # 2. Add the Final Test Input (This is what the model needs to solve)
        test_input = task['test'][0]['input']
        sequence.append(TEST_TOKEN)
        sequence.extend(self._flatten_grid(test_input))
        
        # 3. Add the Target Output (What we train the model to predict)
        test_output = task['test'][0]['output']
        sequence.append(OUTPUT_TOKEN)
        
        # --- Create x_input and y_target ---
        # x_input sees everything UP TO the OUTPUT_TOKEN for the test set
        x_input = sequence.copy()
        
        # y_target is the exact same sequence, but we append the actual answer at the end
        y_target = sequence.copy()
        y_target.extend(self._flatten_grid(test_output))
        
        # Pad sequences to max_seq_len so PyTorch can batch them
        if len(x_input) > self.max_seq_len:
            x_input = x_input[:self.max_seq_len]
        else:
            x_input += [PAD_TOKEN] * (self.max_seq_len - len(x_input))
            
        if len(y_target) > self.max_seq_len:
            y_target = y_target[:self.max_seq_len]
        else:
            y_target += [PAD_TOKEN] * (self.max_seq_len - len(y_target))

        return torch.tensor(x_input, dtype=torch.long), torch.tensor(y_target, dtype=torch.long)

# --- Test the Parser ---
if __name__ == "__main__":
    dataset = ARCDataset()
    x, y = dataset[0]
    
    print("\n--- Testing Tokenizer Output ---")
    print(f"Max Sequence Length: {dataset.max_seq_len}")
    print(f"x_input shape: {x.shape}")
    print(f"y_target shape: {y.shape}")
    
    # Show the first 50 tokens of the first task to verify
    print("\nFirst 50 tokens of Task 0 (x_input):")
    print(x[:50].numpy())
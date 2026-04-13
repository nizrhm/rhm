import torch
import time
from torch.utils.data import DataLoader
from model import SelfEvolvingTRM
from arc_dataset import ARCDataset
from tqdm import tqdm

def main():
    print("Initializing Phase-III Inference Engine...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Selected: {device.type.upper()}")
    
    # 1. Initialize Model with exact training parameters
    max_seq = 1024
    vocab_size = 15
    
    model = SelfEvolvingTRM(
        vocab_size=vocab_size, 
        d_model=256,    
        d_ff=1024,      
        n_layers=4, 
        max_seq_len=max_seq
    )
    
    # Load the weights
    weights_path = "trm_arc_weights_final.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Switch to evaluation mode
    
    print(f"Successfully loaded weights from {weights_path}")

    # 2. Setup Dataset
    # We test on the training set first to verify it can memorize exact solutions.
    # Later, you will swap this for your validation/test set.
    dataset = ARCDataset(max_seq_len=max_seq)
    
    # Batch size 1 is standard for measuring precise inference time per task
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    exact_matches = 0
    total_tasks = 0
    total_time = 0.0
    
    print("Starting Exact Match Evaluation...")
    
    with torch.no_grad(): # Disable gradient tracking for inference
        for x_input, y_target in tqdm(loader, desc="Evaluating Tasks"):
            x_input, y_target = x_input.to(device), y_target.to(device)
            
            # Start timer
            start_time = time.time()
            
            # AMP for faster inference
            with torch.amp.autocast('cuda'):
                final_logits, _, _, _ = model(
                    x_input, 
                    past_z=None, 
                    max_reasoning_steps=8, 
                    max_refinement_steps=4, 
                    early_exit_threshold=1e-4
                )
            
            # End timer
            end_time = time.time()
            total_time += (end_time - start_time)
            total_tasks += 1
            
            # --- EXACT MATCH LOGIC ---
            predictions = torch.argmax(final_logits, dim=-1)
            active_mask = (y_target != 10) # Ignore padding
            
            # A task is only an exact match if EVERY active token matches the target
            is_exact_match = torch.all(predictions[active_mask] == y_target[active_mask]).item()
            
            if is_exact_match:
                exact_matches += 1

    # Metrics
    em_percentage = (exact_matches / total_tasks) * 100
    avg_time_ms = (total_time / total_tasks) * 1000
    
    print("\n" + "="*40)
    print(" PHASE-III EVALUATION RESULTS")
    print("="*40)
    print(f"Total Tasks Evaluated : {total_tasks}")
    print(f"Exact Matches         : {exact_matches} / {total_tasks}")
    print(f"Exact Match Rate      : {em_percentage:.2f} %")
    print(f"Avg Inference Time    : {avg_time_ms:.2f} ms / task")
    print("="*40)

if __name__ == "__main__":
    main()
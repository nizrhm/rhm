import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

def train_arc_trm(model, train_loader, device, epochs=50, lr=0.0003):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine Annealing scheduler smoothly decays the LR to 0 over the 100 epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss(ignore_index=10) 
    discount_factor = 0.5 
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs}")
        
        for batch_idx, (x_input, y_target) in enumerate(pbar):
            x_input, y_target = x_input.to(device), y_target.to(device)
            optimizer.zero_grad(set_to_none=True) 
            
            with torch.amp.autocast('cuda'):
                final_logits, logits_trajectory, _, _ = model(
                    x_input, 
                    past_z=None, # Memory is still OFF for the baseline
                    max_reasoning_steps=2, 
                    max_refinement_steps=4, 
                    early_exit_threshold=1e-4
                )
                
                loss = 0
                for step, step_logits in enumerate(logits_trajectory):
                    step_loss = criterion(step_logits.view(-1, step_logits.size(-1)), y_target.view(-1))
                    weight = discount_factor ** (len(logits_trajectory) - 1 - step)
                    loss += step_loss * weight

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(final_logits.detach().float(), dim=-1)
            active_mask = (y_target != 10)
            
            batch_correct = ((predictions == y_target) & active_mask).sum().item()
            batch_total = active_mask.sum().item()
            correct_tokens += batch_correct
            total_tokens += batch_total
            
            current_acc = (batch_correct / batch_total * 100) if batch_total > 0 else 0.0
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{current_acc:.1f}%", "LR": f"{scheduler.get_last_lr()[0]:.1e}"})

        # Step the scheduler at the end of the epoch
        scheduler.step()

        epoch_time = time.time() - start_time
        epoch_token_acc = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        
        print(f"--> End of Epoch {epoch+1:03d} | Time: {epoch_time:.2f}s | Avg Loss: {avg_loss:.4f} | Overall Acc: {epoch_token_acc:.2f}%")
        
        # --- AUTOMATIC CHECKPOINTING ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "trm_arc_weights_best.pth")
            print(f"    [!] New Best Model Saved (Loss: {best_loss:.4f})")
        print()

    return model
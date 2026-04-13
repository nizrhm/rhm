import torch
from torch.utils.data import DataLoader
from model import SelfEvolvingTRM
from arc_dataset import ARCDataset
from train_arc import train_arc_trm

def main():
    print("Initializing Phase-III Research Hub: ARC-AGI 100-Epoch Run...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Selected: {device.type.upper()}")
    
    vocab_size = 15  
    max_seq = 1024
    
    train_dataset = ARCDataset(data_dir="./arc_data/training", max_seq_len=max_seq)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    ) 
    
    model = SelfEvolvingTRM(
        vocab_size=vocab_size, 
        d_model=256,    
        d_ff=1024,      
        n_layers=4, 
        max_seq_len=max_seq
    )
    
    model = model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} Million")
    
    # Run for 100 Epochs
    trained_model = train_arc_trm(model, train_loader, device, epochs=50, lr=0.0003)
    
    # Save the final state just in case, though the best state is already saved
    torch.save(trained_model.state_dict(), "trm_arc_weights_final.pth")
    print("100-Epoch Training Complete.")

if __name__ == "__main__":
    main()
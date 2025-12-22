import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: DARWINIAN PRUNING EXPERIMENT (Survival of the Fittest)...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CONFIG: Pure MNIST (630k Params) to start
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    NUM_NEURONS = INPUT_SIZE + OUTPUT_SIZE
    
    input_ids = list(range(784))
    output_ids = list(range(784, 794))
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.1, 
        device=DEVICE
    )
    
    trainer = RealNetTrainer(model, device=DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) # Reduced decay, let pruning handle it
    
    trainer.optimizer = optimizer
    trainer.loss_fn = loss_fn
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    SUBSET_SIZE = 10000 
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    NUM_EPOCHS = 50
    THINKING_STEPS = 10
    PRUNE_THRESHOLD = 0.03 # KILL EVERYTHING WEAK
    
    print(f"Total Connections Start: {model.mask.numel()}")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            inputs_val = data.view(data.size(0), -1).to(DEVICE)
            targets_val = torch.ones(data.size(0), 10, device=DEVICE) * -1.0
            for i, label in enumerate(target):
                targets_val[i, label] = 1.0
                
            loss = trainer.train_batch(inputs_val, targets_val, thinking_steps=THINKING_STEPS)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        
        # --- THE REAPER COMES ---
        # Pruning happens AFTER training the epoch (Sleep/Consolidation phase)
        pruned, dead_total, total = trainer.prune(threshold=PRUNE_THRESHOLD)
        sparsity = (dead_total / total) * 100.0
        
        # ACTIVATE TRUE SPARSITY FOR INFERENCE
        from realnet import SparseRealNet
        sparse_model = SparseRealNet.from_dense(model)
        
        # Swap trainer model to sparse for efficient inference
        original_model = trainer.model
        trainer.model = sparse_model
        trainer.model.eval()
        
        # Test
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE)
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes.cpu() == target).sum().item()
                total_samples += target.size(0)
        
        # SWITCH BACK TO DENSE FOR TRAINING (Optimizer needs dense gradients)
        trainer.model = original_model
        trainer.model.train()
        
        acc = 100.0 * correct / total_samples
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Acc {acc:.2f}% | Dead Synapses: {sparsity:.2f}% ({dead_total}/{total})")

if __name__ == "__main__":
    main()

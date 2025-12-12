import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Fix path depth
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: EXTREME Edge Testing - MNIST 50k Params Challenge...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CHALLENGE: Beat MLP (400k params) with ~50k params.
    # Strategy: Downscale spatial resolution, rely on temporal chaos.
    
    # 28x28 -> 14x14 = 196 Inputs.
    INPUT_SIZE = 196
    OUTPUT_SIZE = 10
    HIDDEN_BUFFER = 24 # Just 24 neurons to think!
    
    NUM_NEURONS = INPUT_SIZE + HIDDEN_BUFFER + OUTPUT_SIZE # Wait, let's keep it simpler.
    # Let's map Input 0-195.
    # Let's map Output 220-229.
    # Total 230.
    
    NUM_NEURONS = 230
    PARAMS = NUM_NEURONS * NUM_NEURONS
    
    print(f"Neurons: {NUM_NEURONS}")
    print(f"Params: {PARAMS} ({PARAMS/1000:.1f}k)")
    print(f"Comparison: ~13% of Standard MLP (400k)")
    
    input_ids = list(range(196))
    output_ids = list(range(220, 230))
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.1, 
        device=DEVICE
    )
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # Data: Resize to 14x14
    transform = transforms.Compose([
        transforms.Resize((14, 14)), # Downscale
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # We need a bit more data to learn this compression
    SUBSET_SIZE = 5000 
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01) # Slightly higher LR
    loss_fn = nn.MSELoss()
    trainer.optimizer = optimizer
    trainer.loss_fn = loss_fn
    
    NUM_EPOCHS = 10
    THINKING_STEPS = 15 # More time to think because brain is small
    
    print("Training...")
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
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE)
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes.cpu() == target).sum().item()
                total += target.size(0)
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Acc {acc:.2f}%")

if __name__ == "__main__":
    main()

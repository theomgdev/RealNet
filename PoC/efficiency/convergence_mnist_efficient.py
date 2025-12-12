import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: Edge Testing - Efficient MNIST...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EFFICIENT CONFIG
    # Inputs: 784
    # Outputs: 10
    # Extra Chaos Neurons: 6
    # Total: 800 Neurons
    
    NUM_NEURONS = 800
    input_ids = list(range(784))
    output_ids = list(range(790, 800)) # Last 10 neurons
    
    print(f"Neurons: {NUM_NEURONS}")
    print(f"Params: {NUM_NEURONS*NUM_NEURONS/1e6:.2f}M")
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.1, # Use dropout here for regularization
        device=DEVICE
    )
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Subset
    SUBSET_SIZE = 2000
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(500))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Custom Loop using Trainer internals
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = nn.MSELoss()
    trainer.optimizer = optimizer
    trainer.loss_fn = loss_fn
    
    NUM_EPOCHS = 5
    THINKING_STEPS = 5
    
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
        
        # Test
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

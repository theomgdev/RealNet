import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

# Add parent directory to path to import 'realnet' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from realnet import RealNet, RealNetTrainer

def main():
    print("Initializing RealNet 2.0: MNIST Large Scale Demo...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # --- 1. Configuration for Full Scale ---
    # MNIST images are 28x28 = 784 pixels.
    # Output classes: 0-9 (10 classes).
    # We DO NOT downscale inputs.
    
    # We need at least 784 neurons just to receive the input.
    # We need 10 neurons for output.
    # Plus some "hidden" neurons for chaos and thinking.
    
    # Let's map pixels 0-783 directly to neurons 0-783.
    # Let's map outputs to the LAST 10 neurons.
    # Let's add ~230 extra neurons for pure processing (Total ~1024).
    
    NUM_NEURONS = 1024
    print(f"Total Neurons: {NUM_NEURONS} (Full Connected Matrix: {NUM_NEURONS}x{NUM_NEURONS} = {NUM_NEURONS**2/1e6:.2f} Million Params)")
    
    input_ids = list(range(784)) # Neurons 0 to 783
    output_ids = list(range(NUM_NEURONS - 10, NUM_NEURONS)) # Last 10 neurons
    
    # --- 2. Initialize Model ---
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        device=DEVICE
    )
    
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # --- 3. Prepare MNIST Dataset ---
    # We normalize to [-1, 1] for RealNet preference.
    # Standard MNIST is [0, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # (x - 0.5) / 0.5 -> [-1, 1]
    ])
    
    print("Downloading/Loading MNIST...")
    # Download to a local data folder
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Use a subset for faster demonstration if needed, but let's try a reasonable chunk.
    # Full MNIST is 60000. Let's start with 2000 samples to prove convergence without waiting hours.
    # But batching should be efficient.
    
    SUBSET_SIZE = 2000
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(500))
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {SUBSET_SIZE} samples. Testing on 500 samples.")
    
    # --- 4. Custom Training Loop (Adapter for RealNet) ---
    # RealNet expects specific targets (values for output neurons).
    # MNIST gives class indices (0-9). We need One-Hot Encoding (-1, 1).
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # We assign custom optimizer/loss to trainer (or use trainer's internals manually for loop control)
    trainer.optimizer = optimizer
    trainer.loss_fn = loss_fn
    
    NUM_EPOCHS = 5
    THINKING_STEPS = 5 # Can increase if needed, start small for speed
    
    print("\nStarting Training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: (Batch, 1, 28, 28) -> Flatten -> (Batch, 784)
            inputs_val = data.view(data.size(0), -1).to(DEVICE)
            
            # Prepare Targets: One-Hot (-1 for non-target, 1 for target)
            targets_val = torch.ones(data.size(0), 10, device=DEVICE) * -1.0
            for i, label in enumerate(target):
                targets_val[i, label] = 1.0
                
            # Train Step using Trainer
            loss = trainer.train_batch(inputs_val, targets_val, thinking_steps=THINKING_STEPS)
            total_loss += loss
            
            # Calculate Accuracy (for monitoring)
            with torch.no_grad():
                # We need to re-run predict or peek inside train_batch results.
                # Since train_batch returns only loss, we can't get acc easily without re-running.
                # For efficiency, let's just train. We check acc at end of epoch.
                pass

        avg_loss = total_loss / len(train_loader)
        
        # Validation Accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE)
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                
                # Preds: (Batch, 10). Argmax gives predicted class.
                predicted_classes = torch.argmax(preds, dim=1)
                test_correct += (predicted_classes.cpu() == target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100.0 * test_correct / test_total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Test Acc: {test_acc:.2f}%")

    print("\nTraining Complete.")
    
    # --- 5. Verify Single Example ---
    print("\nVerifying Single Example (Wait and Think)...")
    data, target = test_dataset[0]
    inputs_val = data.view(1, -1).to(DEVICE)
    true_label = target
    
    print(f"Input: Image of digit '{true_label}'")
    
    preds = trainer.predict(inputs_val, thinking_steps=10) # Give it more time to think now
    pred_label = torch.argmax(preds, dim=1).item()
    
    print(f"RealNet Thought Process Output: {preds.cpu().numpy()}")
    print(f"Predicted: {pred_label}")
    print(f"Result: {'SUCCESS' if pred_label == true_label else 'FAIL'}")

if __name__ == "__main__":
    main()

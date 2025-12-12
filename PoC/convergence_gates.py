import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: The Impossible XOR (Zero-Hidden)...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # THE REVOLUTION
    # Traditional AI says: You need hidden layers for XOR.
    # RealNet says: You just need time.
    
    # Neuron 3, 4: Chaos Buffer
    
    NUM_NEURONS = 5
    INPUT_IDS = [0, 1]
    OUTPUT_ID = [2]
    
    print(f"Neurons: {NUM_NEURONS} (25 Parameters)")
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=INPUT_IDS, 
        output_ids=OUTPUT_ID, 
        pulse_mode=True, 
        device=DEVICE
    )
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # XOR Data
    data = [
        (-1.0, -1.0, -1.0),
        (-1.0,  1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0, -1.0),
    ]
    
    # Augment for batching
    inputs_list = []
    targets_list = []
    for _ in range(100):
        for row in data:
            inputs_list.append([row[0], row[1]])
            targets_list.append([row[2]])
            
    inputs_val = torch.tensor(inputs_list, device=DEVICE)
    targets_val = torch.tensor(targets_list, device=DEVICE)

    # 3 Neurons is REALLY tight. We might need a bit more thinking time to propagate signals.
    # Let's give it 15 steps.
    print("Training...")
    history = trainer.fit(inputs_val, targets_val, epochs=200, batch_size=16, thinking_steps=15)

    print(f"Final Loss: {history[-1]:.6f}")

    print("\nVerifying Truth Table:")
    print(f"{'A':>6} {'B':>6} | {'XOR (Pred)':>12} | {'Logic'}")
    print("-" * 40)
    
    test_data = torch.tensor([[-1.0,-1.0], [-1.0,1.0], [1.0,-1.0], [1.0,1.0]], device=DEVICE)
    preds = trainer.predict(test_data, thinking_steps=15)
    
    for i in range(4):
        a = test_data[i][0].item()
        b = test_data[i][1].item()
        out = preds[i].item()
        logic = "1" if out > 0 else "0"
        print(f"{a:>6.1f} {b:>6.1f} | {out:>12.4f} | {logic}")

if __name__ == "__main__":
    main()

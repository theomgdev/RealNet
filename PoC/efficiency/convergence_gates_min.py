import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: Edge Testing - Minimal XOR...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # XOR NEEDS HIDDEN DIMENSIONS (Or Temporal Folding)
    # Inputs: 2
    # Output: 1
    # Hidden: Let's try just 2 extra neurons.
    # Total: 5 Neurons.
    
    NUM_NEURONS = 5
    INPUT_IDS = [0, 1]
    OUTPUT_ID = [2] # Just XOR output
    
    print(f"Neurons: {NUM_NEURONS} (Input: 2, Output: 1, Buffer: 2)")
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=INPUT_IDS, 
        output_ids=OUTPUT_ID, 
        pulse_mode=True, 
        dropout_rate=0.0,
        device=DEVICE
    )
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # XOR Data
    # A, B -> XOR
    data = [
        (-1.0, -1.0, -1.0),
        (-1.0,  1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0, -1.0),
    ]
    
    inputs_list = []
    targets_list = []
    for _ in range(50):
        for row in data:
            inputs_list.append([row[0], row[1]])
            targets_list.append([row[2]])
            
    inputs_val = torch.tensor(inputs_list, device=DEVICE)
    targets_val = torch.tensor(targets_list, device=DEVICE)

    print("Training...")
    trainer.fit(inputs_val, targets_val, epochs=200, batch_size=8, thinking_steps=10)

    print("\nTest Result:")
    test_data = torch.tensor([[-1.0,-1.0], [-1.0,1.0], [1.0,-1.0], [1.0,1.0]], device=DEVICE)
    preds = trainer.predict(test_data, thinking_steps=10)
    
    for i in range(4):
        print(f"In: {test_data[i].tolist()} -> Out: {preds[i].item():.4f}")

if __name__ == "__main__":
    main()

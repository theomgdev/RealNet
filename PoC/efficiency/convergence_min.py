import torch
import sys
import os

# Add parent directory to path to import 'realnet' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: Edge Testing - Minimal Identity...")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # EXTREME MINIMIZATION
    # We try with ONLY 2 neurons.
    # Neuron 0: Input & Output? To strictly test mapping, let's use:
    # Neuron 0: Input
    # Neuron 1: Output
    # No hidden neurons.
    
    NUM_NEURONS = 4 
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    print(f"Neurons: {NUM_NEURONS}")
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=[INPUT_ID], 
        output_ids=[OUTPUT_ID], 
        pulse_mode=True, 
        dropout_rate=0.0, # No dropout for such tiny net
        device=DEVICE
    )
    
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # Data: Identity
    inputs_val = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    targets_val = inputs_val
    
    # Augment
    inputs_val = inputs_val.repeat(100, 1)
    targets_val = targets_val.repeat(100, 1)

    print("Training...")
    trainer.fit(inputs_val, targets_val, epochs=50, batch_size=8, thinking_steps=5)

    print("\nTest Result:")
    test_inputs = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    preds = trainer.predict(test_inputs, thinking_steps=5)
    
    for i in range(len(test_inputs)):
        print(f"In: {test_inputs[i].item()} -> Out: {preds[i].item():.4f}")

if __name__ == "__main__":
    main()

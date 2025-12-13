import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer

def main():
    print("RealNet 2.0: The Atomic Identity...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ATOMIC UNIT OF CHAOS
    # To have a stable StepNorm distribution, we need at least 4 neurons.
    # 1 Input, 1 Output, 2 Chaos Buffers.
    
    NUM_NEURONS = 4
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=[INPUT_ID], 
        output_ids=[OUTPUT_ID], 
        pulse_mode=True, 
        dropout_rate=0.0,
        device=DEVICE
    )
    trainer = RealNetTrainer(model, device=DEVICE)
    
    # Data
    inputs_val = torch.randint(0, 2, (100, 1)).float() * 2 - 1 
    targets_val = inputs_val

    print("Training...")
    trainer.fit(inputs_val, targets_val, epochs=50, batch_size=8, thinking_steps=5)

    print("\nTest Result:")
    test_inputs = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    preds = trainer.predict(test_inputs, thinking_steps=5)
    
    for i in range(len(test_inputs)):
        print(f"In: {test_inputs[i].item()} -> Out: {preds[i].item():.4f}")

if __name__ == "__main__":
    main()

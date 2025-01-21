import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel

# Load the first part of the model
model_path = "gpt2_part1_taher.pth"

class Part1Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use the GPT2LMHeadModel to include the transformer attribute
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        # Retain only the first half of the transformer blocks
        self.model.transformer.h = torch.nn.ModuleList(
            list(self.model.transformer.h)[:len(self.model.transformer.h) // 2]
        )
        # Load the state dictionary for the first part
        self.model.load_state_dict(torch.load(model_path))

    def forward(self, x):
        # Forward pass through the model
        return self.model(x)[0]

# Instantiate and set the model to evaluation mode
model_part1 = Part1Model()
model_part1.eval()

# Flask app to send output to the second laptop
app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    # Receive input tensor
    input_data = torch.tensor(request.json['input'])
    with torch.no_grad():
        # Process the input through the first part of the model
        output = model_part1(input_data)
    
    # Send the output as JSON
    return jsonify({"output": output.tolist()})

if __name__ == '__main__':
    app.run(host='192.168.0.102', port=5000)  # Use your local IP and port

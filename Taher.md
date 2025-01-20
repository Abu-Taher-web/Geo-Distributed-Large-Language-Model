Given your hardware constraints (8 GB RAM, 2 GB VRAM, Core i5 processor), running large LLMs will be challenging, but you can work with smaller or quantized models. Here's a step-by-step workflow to split an LLM across three laptops and run inference collaboratively.

---

### **Step-by-Step Workflow for Splitting an LLM Across Laptops**

#### **Step 1: Choose a Suitable Model**
For your hardware, I recommend these models that can be split and run across devices:

1. **Mistral 7B (quantized 4-bit)** – Good performance with reduced memory footprint (~5-7 GB).
2. **LLaMA 2 (7B, quantized 4-bit)** – Open-source and well-optimized for splitting.
3. **GPT-J 6B (quantized)** – Manageable size with good performance on low-end GPUs.
4. **Phi-2 (2.7B)** – Lightweight and ideal for lower hardware capabilities.
5. **GPT-2 (1.5B)** – A smaller but still capable model for experimentation.

---

#### **Step 2: Install Required Software**
On all three laptops, install:

- **Python (>=3.9)**  
- **PyTorch (for CPU/GPU depending on capability)**  
- **Transformers Library (Hugging Face)**  
- **BitsAndBytes (for quantization)**  
- **MPI (Message Passing Interface, for distributed computing)**  
- **Ray or Deepspeed (for distributed model inference)**  

```bash
pip install torch transformers accelerate bitsandbytes mpi4py ray deepspeed
```

---

#### **Step 3: Quantize and Split the Model**
Since your hardware is limited, quantization is crucial to reduce memory load. Use Hugging Face's `bitsandbytes` to load a quantized model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mistralai/Mistral-7B-v0.1"

# Load quantized model (4-bit for low memory)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model in chunks for splitting
torch.save(model.state_dict(), "model_split.pth")
```

You can manually split the model checkpoint into 3 parts (manually or using Python's `shutil` module).

---

#### **Step 4: Distribute Model Parts Across Laptops**
Split model weights across 3 machines, for example:

1. **Laptop 1:** Embedding + Attention layers (first part)
2. **Laptop 2:** Middle layers
3. **Laptop 3:** Final layers + Output

Copy the respective model chunks to each laptop using:

```bash
scp model_part1.pth user@laptop1:/path/to/store
scp model_part2.pth user@laptop2:/path/to/store
scp model_part3.pth user@laptop3:/path/to/store
```

---

#### **Step 5: Run Distributed Inference**
Use **Ray** or **MPI** to distribute inference across the laptops. Example with MPI:

On Laptop 1, run the master script:

```bash
mpirun -np 3 --host laptop1,laptop2,laptop3 python distributed_inference.py
```

Sample code for distributed inference (`distributed_inference.py`):

```python
from mpi4py import MPI
import torch
from transformers import AutoTokenizer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

if rank == 0:
    # Laptop 1 - Load and process first part
    model_part1 = torch.load("model_part1.pth")
    comm.send(model_part1, dest=1)
elif rank == 1:
    # Laptop 2 - Receive, process, and forward
    model_part1 = comm.recv(source=0)
    model_part2 = torch.load("model_part2.pth")
    comm.send(model_part2, dest=2)
elif rank == 2:
    # Laptop 3 - Final processing
    model_part2 = comm.recv(source=1)
    model_part3 = torch.load("model_part3.pth")
    print("Inference completed")
```

---

#### **Step 6: Synchronization and Communication**
Use efficient networking via **Ethernet** or a local wireless network to ensure smooth data transfer between machines.

---

#### **Step 7: Optimize Performance**
- Use **quantization (4-bit/8-bit)** to reduce memory load.
- Load parts of the model to the GPU when available.
- Use lightweight frameworks like **GGUF format** for efficient loading on CPU.

---

### **Challenges and Considerations**
1. **Latency:** Communication overhead can slow down inference.
2. **Memory Limitations:** Keep model parts small enough to fit within the available RAM/VRAM.
3. **Parallelism:** Ensure correct ordering of layers to maintain inference consistency.
4. **Hardware Compatibility:** Ensure all laptops have compatible Python environments.

---

#Split into two part

---

### **Step 1: Load the GPT-2 Model**
First, install the required dependencies if you haven't already:

```bash
pip install torch transformers
```

Now, load the model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the entire model for reference (optional)
torch.save(model.state_dict(), "gpt2_full.pth")
```

---

### **Step 2: Split the Model into Two Parts**
GPT-2 consists of transformer blocks (e.g., 12 for GPT-2 small). We can split them into two halves.

```python
# Get all transformer blocks
transformer_blocks = model.transformer.h

# Split model into two halves
midpoint = len(transformer_blocks) // 2
part1_blocks = transformer_blocks[:midpoint]
part2_blocks = transformer_blocks[midpoint:]

# Create two separate model parts
model_part1 = torch.nn.Sequential(*part1_blocks)
model_part2 = torch.nn.Sequential(*part2_blocks)

# Save the two parts separately
torch.save(model_part1.state_dict(), "gpt2_part1.pth")
torch.save(model_part2.state_dict(), "gpt2_part2.pth")

print("Model successfully split into two parts.")
```

---

### **Step 3: Transfer the Model Parts to Different Laptops**
After splitting, transfer each part to the respective machine using SCP:

```bash
scp "E:\GitHub_Desktop\Geo-Distributed-Large-Language-Model\gpt2_part1_taher.pth" user@laptop1:/path/to/store/
scp "E:\GitHub_Desktop\Geo-Distributed-Large-Language-Model\gpt2_part2_rabiul.pth" "Robiul Hasan@192.168.0.103:'D:\University of Oulu\Last Education\1st Year\3rd Period\DS'"
```

Replace `user`, `laptop1/laptop2`, and `/path/to/store/` with your actual details.

---

### **Step 4: Load and Run Inference Separately**
On each laptop, load and process its part of the model.

**Laptop 1 (part 1):**
```python
import torch
from transformers import AutoTokenizer

model_part1 = torch.load("gpt2_part1.pth")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Process through first half
output_part1 = model_part1(input_ids)
torch.save(output_part1, "output_part1.pth")
```

**Laptop 2 (part 2):**
```python
import torch

model_part2 = torch.load("gpt2_part2.pth")
output_part1 = torch.load("output_part1.pth")

# Process through second half
final_output = model_part2(output_part1)
print("Final output:", final_output)
```

---

# How to send inference data from one laptop to another.
If the GPT-2 model is split in such a way that part 2 depends on the output of part 1 (i.e., the output of part 1 is the input for part 2), you will need a way to synchronize the flow between both parts during inference. This means that the model can’t perform the inference entirely in parallel without some level of coordination.

Here's how you can approach this:

### 1. **Sequential Dependency Between Parts**:
   - **Laptop 1 (Master)**: Runs **Part 1** of the model and performs inference on the input data. Once the inference is complete, it sends the output to **Laptop 2**.
   - **Laptop 2 (Worker)**: Receives the output from **Laptop 1**, uses it as input for **Part 2**, and performs inference.
   - **Result Collection**: After **Laptop 2** finishes, it sends the final result back to **Laptop 1** for further processing or display.

### 2. **Setup for Communication Between Laptops**:
   Since the two parts are sequential, you’ll need a way to pass data between the two laptops.

   - You can use **Sockets** or **HTTP Requests** for communication between laptops:
     - **Sockets**: Create a socket server on Laptop 2 and a client on Laptop 1. Once Laptop 1 completes its part of the inference, it sends the output via the socket to Laptop 2.
     - **HTTP Requests**: You can set up a lightweight server (using Flask or FastAPI) on Laptop 2 to receive the output from Laptop 1 as a POST request and then proceed with the inference on part 2.

### 3. **Implementation Example (using Sockets)**:
   Here's a simple example using Python’s `socket` library to communicate between the two laptops.

   **Laptop 1 (Master - Sends data to Laptop 2):**
   ```python
   import socket
   import torch
   from transformers import GPT2Model, GPT2Tokenizer

   # Load Part 1 of GPT-2 model
   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   model_part_1 = GPT2Model.from_pretrained("gpt2")

   # Prepare the input
   input_text = "This is the input for part 1."
   inputs = tokenizer(input_text, return_tensors="pt")

   # Perform inference on part 1
   with torch.no_grad():
       output_part_1 = model_part_1(**inputs)

   # Send output to Laptop 2 via socket
   part_1_output = output_part_1.last_hidden_state.numpy()
   
   # Connect to Laptop 2 (Worker) via socket
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   sock.connect(('laptop_2_ip', 12345))  # Replace with Laptop 2's IP
   sock.sendall(part_1_output.tobytes())
   sock.close()
   ```

   **Laptop 2 (Worker - Receives and processes data):**
   ```python
   import socket
   import torch
   from transformers import GPT2LMHeadModel, GPT2Tokenizer

   # Load Part 2 of GPT-2 model
   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   model_part_2 = GPT2LMHeadModel.from_pretrained("gpt2")

   # Set up the socket server
   server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   server_socket.bind(('0.0.0.0', 12345))  # Listen on all interfaces, port 12345
   server_socket.listen(1)
   print("Waiting for connection...")
   client_socket, _ = server_socket.accept()

   # Receive output from Laptop 1
   part_1_output = client_socket.recv(4096)  # Adjust buffer size if needed
   part_1_output = torch.from_numpy(np.frombuffer(part_1_output, dtype=np.float32))

   # Perform inference on part 2
   with torch.no_grad():
       output_part_2 = model_part_2(inputs_embeds=part_1_output)

   print("Inference on part 2 complete.")
   client_socket.close()
   ```

### 4. **HTTP Setup with Flask (Alternative)**:
   If you prefer HTTP requests instead of sockets, you can use Flask to create a server on Laptop 2 that listens for POST requests from Laptop 1.

   **Laptop 2 (Worker - Flask server)**:
   ```python
   from flask import Flask, request, jsonify
   import torch
   from transformers import GPT2LMHeadModel

   app = Flask(__name__)

   model_part_2 = GPT2LMHeadModel.from_pretrained("gpt2")

   @app.route('/process', methods=['POST'])
   def process():
       data = request.get_json()
       part_1_output = torch.tensor(data['output_part_1'])

       with torch.no_grad():
           output_part_2 = model_part_2(inputs_embeds=part_1_output)
       
       return jsonify({'result': output_part_2.tolist()})

   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5000)
   ```

   **Laptop 1 (Master - Sends data via HTTP)**:
   ```python
   import requests
   import torch
   from transformers import GPT2Model, GPT2Tokenizer

   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   model_part_1 = GPT2Model.from_pretrained("gpt2")

   input_text = "This is the input for part 1."
   inputs = tokenizer(input_text, return_tensors="pt")

   with torch.no_grad():
       output_part_1 = model_part_1(**inputs)

   part_1_output = output_part_1.last_hidden_state.numpy().tolist()

   # Send output to Laptop 2 via HTTP
   response = requests.post('http://laptop_2_ip:5000/process', json={'output_part_1': part_1_output})
   result = response.json()
   print("Part 2 Output:", result['result'])
   ```

### Summary:
- Since **Part 2** depends on **Part 1**, you'll need to transfer the output from **Laptop 1 (Master)** to **Laptop 2 (Worker)** for further processing.
- You can use **Sockets** or **HTTP** for this data exchange.
- The flow will be sequential, but you still need network communication between the laptops to pass data back and forth.

Let me know if you need more help with the code or network setup!

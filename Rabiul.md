To set up SSH on the other laptop (Windows), follow these steps:

---

### **1. Install OpenSSH Server (if not already installed)**

1. Open **PowerShell as Administrator** and run:  
   ```powershell
   Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
   ```
   If it shows "Not Present," install it with:  
   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   ```

---

### **2. Start the SSH Service**

1. Open PowerShell as Administrator and run:  
   ```powershell
   Start-Service sshd
   ```

2. Enable SSH to start automatically on boot:  
   ```powershell
   Set-Service -Name sshd -StartupType Automatic
   ```

---

### **3. Allow SSH Through Windows Firewall**

1. Run this command to allow SSH traffic:  
   ```powershell
   New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
   ```

---

### **4. Find the Laptop's IP Address**

Run the following command to get the IP:  
   ```powershell
   ipconfig
   ```
Look for the `IPv4 Address` under the active network connection (e.g., Wi-Fi or Ethernet).

---

### **5. Test SSH Connection**

From your main laptop, try:  
```powershell
ssh username@<other-laptop-ip>
```

Replace `username` with the actual Windows username of the other laptop.

---
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"

# Load GPT-2 model (quantized for low memory)
#model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save model in chunks for splitting (if necessary)
torch.save(model.state_dict(), "gpt2_model_split.pth")
# Get the state dictionary of the model
state_dict = model.state_dict()

# Print out the keys to see the model layers (for splitting)
for key in state_dict.keys():
    print(key)

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
torch.save(model_part1.state_dict(), "gpt2_part1_taher.pth")
torch.save(model_part2.state_dict(), "gpt2_part2_rabiul.pth")

print("Model successfully split into two parts.")


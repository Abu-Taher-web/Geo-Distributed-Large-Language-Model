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

--------------------------------------
# Troubleshooting "Request Timed Out" When Pinging a Laptop on the Same Network

If you are unable to ping one laptop from another on the same network and receive a `"Request timed out"` message, it typically indicates a configuration or network issue. Follow the steps below to troubleshoot and resolve the problem.

---

## **Steps to Resolve the Issue**

### **1. Check the Firewall Settings**
The Windows Firewall may be blocking ICMP (ping) requests:
1. On the target laptop (the one being pinged), open the **Start menu**, type `Windows Security`, and open it.
2. Navigate to **Firewall & network protection > Advanced settings**.
3. In the left pane, click **Inbound Rules**.
4. Locate the rule: **File and Printer Sharing (Echo Request - ICMPv4-In)**.
5. If this rule is disabled, right-click it and select **Enable Rule**.
6. Repeat the same for **ICMPv6-In** if you want to allow IPv6 ping.

---

### **2. Check Network Connection**
1. Ensure both laptops are connected to the same network (Wi-Fi or Ethernet).
2. Run the following command on both laptops to verify thei
---------------------------------------
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


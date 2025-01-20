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

Let me know if you encounter any issues!

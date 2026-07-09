import pty
import os
import sys
import time
import select

def run_remote(command):
    password = "hardy-urban-boxlike"
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", "9022", "lcorreia@erdos.ex.ac.uk", command]
    
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp("ssh", ssh_cmd)
    else:
        output = b""
        password_sent = False
        start_time = time.time()
        while time.time() - start_time < 20:
            r, w, e = select.select([fd], [], [], 0.5)
            if fd in r:
                try:
                    chunk = os.read(fd, 1024)
                    if not chunk: break
                    output += chunk
                    if b"password:" in chunk.lower() and not password_sent:
                        os.write(fd, (password + "\n").encode())
                        password_sent = True
                except OSError:
                    break
            if os.waitpid(pid, os.WNOHANG) != (0, 0):
                break
        return output.decode(errors='ignore')

print("--- Erdos Memory ---")
print(run_remote("free -h"))
print("\n--- Erdos Load ---")
print(run_remote("uptime"))

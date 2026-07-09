import pty
import os
import sys
import time
import select

def run_remote(command):
    password = "hardy-urban-boxlike"
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-p", "9022", "lcorreia@erdos.ex.ac.uk", command]
    
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

print("Stopping worker loop and screens...")
print(run_remote("pkill -u lcorreia -9 -f 'run_worker_loop.sh'"))
print(run_remote("pkill -u lcorreia -9 -f 'SCREEN'"))
print(run_remote("pkill -u lcorreia -9 -f 'dash'"))

print("\nChecking process counts after cleanup...")
print(run_remote("ps -u lcorreia -L | wc -l"))

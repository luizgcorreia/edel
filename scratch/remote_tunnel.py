import os
import pty
import subprocess
import time
import sys

def main():
    pw = "hardy-urban-boxlike"
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('ssh', ['ssh', '-N', '-p', '9022', '-L', '8050:127.0.0.1:8050', '-o', 'PreferredAuthentications=password', '-o', 'StrictHostKeyChecking=no', 'lcorreia@erdos.ex.ac.uk'])
    else:
        # Give it a moment to prompt
        time.sleep(1)
        buf = b""
        while b"password" not in buf.lower() and b"failed" not in buf.lower():
            try:
                r = os.read(fd, 1024)
                if not r: break
                buf += r
            except: break
        
        if b"password" in buf.lower():
            os.write(fd, (pw + "\n").encode())
            print("Password sent to tunnel ssh.")
        
        # Keep reading or sleep
        print("Tunnel running...")
        while True:
            try:
                r = os.read(fd, 1024)
                if not r:
                    print("Connection closed.")
                    break
            except OSError:
                break
            time.sleep(0.5)

if __name__ == "__main__":
    main()

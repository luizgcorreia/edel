import os
import pty
import time
import sys

def main():
    pw = "hardy-urban-boxlike"
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('ssh', ['ssh', '-p', '9022', '-L', '8050:localhost:8050', '-N', '-o', 'PreferredAuthentications=password', '-o', 'StrictHostKeyChecking=no', 'lcorreia@erdos.ex.ac.uk'])
    else:
        time.sleep(1.5)
        buf = b""
        while b"password" not in buf.lower() and b"failed" not in buf.lower():
            try:
                r = os.read(fd, 1024)
                if not r: break
                buf += r
            except: break
        
        if b"password" in buf.lower():
            os.write(fd, (pw + "\n").encode())
            print("Password sent to SSH tunnel process.")
            time.sleep(1.5)
        else:
            print("Password prompt not found. Buffer:", buf)
        print("Tunnel process running with PID:", pid)

if __name__ == "__main__":
    main()

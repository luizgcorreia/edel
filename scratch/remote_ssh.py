import os
import pty
import subprocess
import time
import sys

def run_ssh(password, cmd):
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('ssh', ['ssh', '-p', '9022', '-o', 'PreferredAuthentications=password', '-o', 'StrictHostKeyChecking=no', 'lcorreia@erdos.ex.ac.uk', cmd])
    else:
        # Give it a moment to prompt
        time.sleep(1)
        # Try to read until password prompt
        buf = b""
        while b"password" not in buf.lower() and b"failed" not in buf.lower():
            try:
                r = os.read(fd, 1024)
                if not r: break
                buf += r
            except: break
        
        if b"password" in buf.lower():
            os.write(fd, (password + "\n").encode())
        
        # Read output
        output = b""
        try:
            while True:
                r = os.read(fd, 1024)
                if not r: break
                output += r
                sys.stdout.buffer.write(r)
                sys.stdout.flush()
        except OSError:
            pass
        return ""

if __name__ == "__main__":
    pw = "hardy-urban-boxlike"
    command = "ps -u lcorreia -f"
    if len(sys.argv) > 1:
        command = sys.argv[1]
    print(run_ssh(pw, command))

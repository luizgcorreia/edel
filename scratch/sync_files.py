import base64
import subprocess
import sys
from pathlib import Path

def sync_file(local_path, remote_path):
    local_path = Path(local_path)
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode('ascii')
    
    remote_cmd = f"python3 -c \"import base64; open('{remote_path}', 'wb').write(base64.b64decode('{encoded}'))\""
    
    proc = subprocess.run(
        ['python3', '/home/correia/edel/scratch/remote_ssh.py', remote_cmd],
        capture_output=True,
        text=True
    )
    if proc.returncode == 0:
        print(f"Successfully synced {local_path} -> {remote_path}")
    else:
        print(f"Failed to sync {local_path}: {proc.stderr} {proc.stdout}")

if __name__ == "__main__":
    sync_file("edel/dashboard/worker.py", "edel/edel/dashboard/worker.py")
    sync_file("scripts/run_dashboard.sh", "edel/scripts/run_dashboard.sh")
    sync_file("scratch/assemble_embeddings.py", "edel/scratch/assemble_embeddings.py")
    sync_file("edel/dashboard/components/job_panel.py", "edel/edel/dashboard/components/job_panel.py")
    sync_file("edel/dashboard/callbacks/experiments.py", "edel/edel/dashboard/callbacks/experiments.py")
    sync_file("edel/experiments/runner.py", "edel/edel/experiments/runner.py")
    sync_file("edel/dashboard/callbacks/config.py", "edel/edel/dashboard/callbacks/config.py")
    sync_file("edel/pipeline/structuring.py", "edel/edel/pipeline/structuring.py")

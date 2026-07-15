"""rp.py — RunPod GPU control for the faced scaling ladder (SDK + SSH).

Ported from riggs/cloud/runpod/rp.py: drives the whole lifecycle with a
RUNPOD_API_KEY (read from ../.env). SSH uses a keypair under runpod/.ssh, injected
to the pod via PUBLIC_KEY (RunPod base images authorize it on boot).

  gpus                  list GPU types + price
  up                    create the A100 pod (waits, prints ssh)
  status <id>           pod state + connection
  deploy <id>           push code, install deps, start the ladder in the background
  logs <id>             tail the remote run log
  fetch <id>            scp artifacts/ back to _remote/
  exec <id> -- <cmd>    run a command over SSH
  down <id>             terminate the pod
  ladder                one-shot: up -> deploy (then poll `logs`, `fetch`, `down`)

Balance note: on-demand pods need a positive RunPod balance; an empty account
returns INSUFFICIENT_BALANCE at `up`/`ladder`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import runpod

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SSH_DIR = HERE / ".ssh"
KEY = SSH_DIR / "id_ed25519"
KNOWN_HOSTS = SSH_DIR / "known_hosts"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA A100 80GB PCIe"
MODELS = ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]
DEPS = ("transformers>=5.5 safetensors huggingface_hub scikit-learn numpy einops "
        "pyyaml matplotlib accelerate")


def load_env():
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not set (put it in D:/face/.env)")
    runpod.api_key = key


def ensure_keypair() -> str:
    SSH_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(KEY), "-q"], check=True)
    return KEY.with_suffix(".pub").read_text().strip()


def _ssh_base(ip, port):
    return ["ssh", "-i", str(KEY), "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", "ConnectTimeout=15", f"root@{ip}"]


def pod_conn(pod):
    for p in (pod.get("runtime") or {}).get("ports") or []:
        if p.get("privatePort") == 22 and p.get("type") == "tcp" and p.get("isIpPublic"):
            return p.get("ip"), p.get("publicPort")
    return None, None


def _get(pid):
    pod = runpod.get_pod(pid)
    return pod.get("pod", pod) if isinstance(pod, dict) else pod


def wait_running(pid, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = _get(pid) or {}
        ip, port = pod_conn(pod)
        if ip and port:
            return pod, ip, port
        print(f"  waiting... status={pod.get('desiredStatus')}")
        time.sleep(10)
    raise TimeoutError(f"pod {pid} not SSH-ready in {timeout}s")


def _scp(port, src, dst, recursive=False):
    cmd = ["scp"] + (["-r"] if recursive else []) + [
        "-i", str(KEY), "-P", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", src, dst]
    return subprocess.run(cmd).returncode


# ---- commands ----
def cmd_gpus(a):
    for g in runpod.get_gpus():
        try:
            d = runpod.get_gpu(g["id"])
            price = (d.get("lowestPrice") or {}).get("minimumBidPrice") or d.get("securePrice")
        except Exception:
            price = None
        if any(k in g["id"] for k in ["A100", "H100", "A6000", "L40", "A40"]):
            print(f"  {('$'+str(price)+'/hr') if price else 'n/a':>12}  {g.get('memoryInGb')}GB  {g['id']}")


def cmd_up(a):
    pub = ensure_keypair()
    print(f"creating pod ({a.gpu}) ...")
    pod = runpod.create_pod(
        name="faced-ladder", image_name=a.image, gpu_type_id=a.gpu, cloud_type=a.cloud,
        support_public_ip=True, start_ssh=True, gpu_count=1,
        container_disk_in_gb=a.disk, volume_in_gb=0, ports="22/tcp",
        env={"PUBLIC_KEY": pub, "HF_TOKEN": os.environ.get("HF_TOKEN", "")})
    pid = pod["id"]
    (REPO / "artifacts").mkdir(exist_ok=True)
    (REPO / "artifacts" / "pod_id.txt").write_text(pid)
    print(f"pod id: {pid}")
    _, ip, port = wait_running(pid)
    print(f"READY  ssh -i {KEY} -p {port} root@{ip}")
    return pid, ip, port


def _pack():
    tar = REPO.parent / "faced_code.tar.gz"
    subprocess.run(["tar", "czf", str(tar), "-C", str(REPO), "faced", "scripts", "runpod",
                    "config/models.yaml", "config/emotions.yaml", "data/prompts", "data/refusal",
                    "data/reference_corpus.jsonl", "requirements.txt"], check=True)
    return tar


def cmd_deploy(a):
    pod = _get(a.pod_id); ip, port = pod_conn(pod)
    if not ip:
        sys.exit("pod not SSH-ready")
    tar = _pack()
    print("pushing code ...")
    _scp(port, str(tar), f"root@{ip}:/workspace/code.tar.gz")
    ssh = _ssh_base(ip, port)
    models = " ".join(a.models)
    script = getattr(a, "script", None) or "runpod/run_ladder.py"
    remote = (
        "cd /workspace && tar xzf code.tar.gz && "
        f"pip install -q -U {DEPS} && "
        f"nohup python {script} "
        f"--models {models} --dtype bfloat16 > /workspace/run.log 2>&1 & "
        "sleep 2 && echo STARTED && head -3 /workspace/run.log")
    subprocess.run(ssh + [remote])
    print(f"\nladder started on {a.pod_id}. Poll: python runpod/rp.py logs {a.pod_id}")


def cmd_logs(a):
    pod = _get(a.pod_id); ip, port = pod_conn(pod)
    subprocess.run(_ssh_base(ip, port) + [
        "tail -n 40 /workspace/run.log; echo '---'; "
        "ls /workspace/artifacts/*.json /workspace/artifacts/*/*.json 2>/dev/null | tail -20; "
        "test -f /workspace/artifacts/DONE.marker && echo '=== DONE ==='"])


def cmd_fetch(a):
    pod = _get(a.pod_id); ip, port = pod_conn(pod)
    dst = REPO / "artifacts" / "_ladder"
    dst.mkdir(parents=True, exist_ok=True)
    _scp(port, f"root@{ip}:/workspace/artifacts", str(dst), recursive=True)
    print(f"fetched -> {dst}")


def cmd_status(a):
    pod = _get(a.pod_id); ip, port = pod_conn(pod)
    print(f"status={pod.get('desiredStatus')} cost/hr={pod.get('costPerHr')} "
          f"ssh={'ready' if ip else 'no'}")


def cmd_exec(a):
    pod = _get(a.pod_id); ip, port = pod_conn(pod)
    sys.exit(subprocess.run(_ssh_base(ip, port) + [" ".join(a.cmd)]).returncode)


def cmd_down(a):
    runpod.terminate_pod(a.pod_id)
    print(f"terminated {a.pod_id}")


def cmd_ladder(a):
    pid, ip, port = cmd_up(a)
    a.pod_id = pid
    cmd_deploy(a)
    print(f"\n=== ladder running on {pid}. Monitor: python runpod/rp.py logs {pid} ===")


def main():
    load_env()
    ap = argparse.ArgumentParser(prog="rp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def pod_opts(p):
        p.add_argument("--gpu", default=GPU)
        p.add_argument("--image", default=IMAGE)
        p.add_argument("--cloud", default="ALL")
        p.add_argument("--disk", type=int, default=300)
        p.add_argument("--models", nargs="+", default=MODELS)

    sub.add_parser("gpus").set_defaults(func=cmd_gpus)
    p = sub.add_parser("up"); pod_opts(p); p.set_defaults(func=cmd_up)
    p = sub.add_parser("ladder"); pod_opts(p); p.set_defaults(func=cmd_ladder)
    p = sub.add_parser("deploy"); p.add_argument("pod_id"); p.add_argument("--models", nargs="+", default=MODELS); p.add_argument("--script", default="runpod/run_ladder.py"); p.set_defaults(func=cmd_deploy)
    for name, fn in [("status", cmd_status), ("logs", cmd_logs), ("fetch", cmd_fetch), ("down", cmd_down)]:
        p = sub.add_parser(name); p.add_argument("pod_id"); p.set_defaults(func=fn)
    p = sub.add_parser("exec"); p.add_argument("pod_id"); p.add_argument("cmd", nargs=argparse.REMAINDER); p.set_defaults(func=cmd_exec)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

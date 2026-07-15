"""Local RunPod orchestrator for the gemma-3 ladder. Careful, staged, teardownable.

    python runpod/launch.py create          # deploy the A100 pod (spends money)
    python runpod/launch.py status <podId>  # pod state + latest log tail from HF
    python runpod/launch.py fetch  <podId>  # download artifacts/ from HF -> local
    python runpod/launch.py stop   <podId>  # TERMINATE the pod (stops billing)

RunPod GraphQL is called via curl (its Cloudflare blocks urllib's user-agent).
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = "fractalnature/faced-runpod"
GPU = "NVIDIA A100 80GB PCIe"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DOCKER = ("bash -c 'pip install -q -U huggingface_hub && "
          "huggingface-cli download " + DATASET + " bootstrap.sh --repo-type dataset "
          "--local-dir /workspace && bash /workspace/bootstrap.sh'")


def envs():
    e = {}
    for line in open(REPO / ".env", encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            e[k.strip()] = v.strip().strip('"').strip("'")
    return e


ENV = envs()


def gql(query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}})
    url = f"https://api.runpod.io/graphql?api_key={ENV['RUNPOD_API_KEY']}"
    r = subprocess.run(["curl", "-s", "--max-time", "60", url,
                        "-H", "Content-Type: application/json", "-d", "@-"],
                       input=payload, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout[:500], "_err": r.stderr[:300]}


def create():
    variables = {"input": {
        "cloudType": "COMMUNITY", "gpuCount": 1, "gpuTypeId": GPU,
        "name": "faced-ladder", "imageName": IMAGE,
        "containerDiskInGb": 300, "volumeInGb": 0,
        "dockerArgs": DOCKER,
        "env": [{"key": "HF_TOKEN", "value": ENV["HF_TOKEN"]}],
    }}
    q = ("mutation deploy($input: PodFindAndDeployOnDemandInput) { "
         "podFindAndDeployOnDemand(input: $input) { id imageName machineId } }")
    d = gql(q, variables)
    print(json.dumps(d, indent=2)[:800])
    pod = (d.get("data") or {}).get("podFindAndDeployOnDemand") or {}
    if pod.get("id"):
        (REPO / "artifacts").mkdir(exist_ok=True)
        (REPO / "artifacts" / "pod_id.txt").write_text(pod["id"])
        print(f"\nPOD ID: {pod['id']}  (saved to artifacts/pod_id.txt)")
    else:
        print("\n!! no pod id — check errors above (GPU availability? try SECURE cloud?)")


def status(pid):
    q = ('query { pod(input:{podId:"%s"}) { id name desiredStatus '
         'runtime { uptimeInSeconds } } }' % pid)
    print(json.dumps(gql(q), indent=2)[:600])
    _hf_tail()


def _hf_tail():
    try:
        from huggingface_hub import hf_hub_download
        for f in ("artifacts/ladder_status.json", "artifacts/ladder_run.log"):
            try:
                p = hf_hub_download(DATASET, f, repo_type="dataset",
                                    local_dir=str(REPO / "_remote"), token=ENV["HF_TOKEN"],
                                    force_download=True)
                txt = open(p, encoding="utf-8").read()
                print(f"\n--- {f} (tail) ---\n" + txt[-1200:])
            except Exception as e:
                print(f"({f}: not available yet — {type(e).__name__})")
    except Exception as e:
        print("hf tail error:", e)


def fetch(pid=None):
    from huggingface_hub import snapshot_download
    p = snapshot_download(DATASET, repo_type="dataset", allow_patterns=["artifacts/*"],
                          local_dir=str(REPO / "_remote"), token=ENV["HF_TOKEN"])
    print("downloaded ->", p)


def stop(pid):
    d = gql('mutation { podTerminate(input:{podId:"%s"}) }' % pid)
    print(json.dumps(d, indent=2)[:400])
    print("terminated" if "errors" not in d else "check errors")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    arg = sys.argv[2] if len(sys.argv) > 2 else \
        (REPO / "artifacts" / "pod_id.txt").read_text().strip() \
        if (REPO / "artifacts" / "pod_id.txt").exists() else None
    {"create": lambda: create(), "status": lambda: status(arg),
     "fetch": lambda: fetch(arg), "stop": lambda: stop(arg)}[cmd]()

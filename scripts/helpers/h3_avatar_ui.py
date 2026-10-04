from __future__ import annotations

import html
import base64
import json
import os
import re
import shutil
import ssl
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "artifacts" / "ui_jobs"
JOBS.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "audio-v2": "/runs/h3/locked/h3-avatar-audio-v2-step1000.pt",
}

STATUS: dict[str, dict] = {}
DEFAULT_CHECKPOINT = "audio-v2"
WARM_ENDPOINT_URL = os.environ.get("H3_WARM_ENDPOINT_URL", "").strip()


def valid_frames(seconds: int) -> int:
    # MiniMax H3 VAE accepts 17n + 5 frames. At 24fps, 243 is ~10.1s.
    target = max(22, int(seconds * 24))
    n = max(1, round((target - 5) / 17))
    return 17 * n + 5


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return proc.stdout


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    def send(context: ssl.SSLContext | None = None) -> dict:
        request = Request(url, data=data, headers=headers, method="POST")
        with urlopen(request, timeout=4 * 60 * 60, context=context) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        result = send()
    except (ssl.SSLCertVerificationError, URLError) as exc:
        is_cert_error = isinstance(exc, ssl.SSLCertVerificationError) or isinstance(
            getattr(exc, "reason", None),
            ssl.SSLCertVerificationError,
        )
        if not is_cert_error:
            raise
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
            result = send(context)
        except Exception:
            context = ssl._create_unverified_context()
            result = send(context)
            result["ssl_warning"] = "Used local-dev unverified HTTPS fallback"
    except HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(body)
            detail = payload.get("detail") or payload.get("error") or body
        except Exception:
            detail = body or str(exc)
        raise RuntimeError(f"warm endpoint failed with HTTP {exc.code}: {detail}") from exc

    if result.get("state") == "error" or result.get("error"):
        raise RuntimeError(result.get("detail") or result.get("error") or "warm endpoint returned an error")
    return result


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = ("--" + match.group("boundary").strip('"')).encode()
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in body.split(boundary):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, _, data = part.partition(b"\r\n\r\n")
        if not data:
            continue
        data = data.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", "replace")
        disp = next((line for line in headers.split("\r\n") if line.lower().startswith("content-disposition:")), "")
        name_match = re.search(r'name="([^"]+)"', disp)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disp)
        if filename_match:
            filename = Path(filename_match.group(1)).name or "reference.png"
            files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


def set_status(status_id: str, **updates) -> None:
    current = STATUS.setdefault(status_id, {})
    current.update(updates)
    current.setdefault("job_id", status_id)
    current["updated_at"] = time.time()
    job_dir = JOBS / status_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "status.json").write_text(json.dumps(current, indent=2), encoding="utf-8")


def generate_job(job_id: str, image_path: Path, fields: dict[str, str]) -> None:
    try:
        duration = int(fields.get("duration", "10"))
        frames = valid_frames(duration)
        steps = int(fields.get("steps", "30"))
        checkpoint_key = fields.get("checkpoint", DEFAULT_CHECKPOINT)
        checkpoint = CHECKPOINTS.get(checkpoint_key, CHECKPOINTS[DEFAULT_CHECKPOINT])
        generate_audio = fields.get("audio", "model") == "model"
        prompt = fields.get("prompt", "").strip() or (
            "The same character faces the camera and speaks with an audible natural voice, "
            "expressive synchronized mouth movement, subtle head motion, and a friendly expression."
        )
        dialogue = fields.get("dialogue", "").strip()
        if dialogue:
            prompt = f'{prompt} Make the character say: "{dialogue}".'
        background = fields.get("background", "").strip()
        if background:
            prompt = f"{prompt} The scene is set {background}."

        job_dir = JOBS / job_id
        ref = job_dir / "reference_512.mp4"
        remote_ref = f"/samples/ui_refs/{job_id}/reference_512.mp4"
        remote_out = f"/generated/ui/{job_id}/generated.mp4"
        container_out = f"/runs{remote_out}"
        local_out = job_dir / "generated.mp4"

        set_status(job_id, state="encoding_reference", detail="Creating 512px reference video", progress=5)
        run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-t", str(duration), "-r", "24",
            "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-an", str(ref),
        ])

        if WARM_ENDPOINT_URL:
            set_status(job_id, state="uploading", detail="Sending reference to warm endpoint", progress=10)
            payload = {
                "checkpoint": checkpoint,
                "reference_video_b64": base64.b64encode(ref.read_bytes()).decode("ascii"),
                "output_path": f"/tmp/h3_ui/{job_id}/generated.mp4",
                "prompt": prompt,
                "num_frames": frames,
                "height": 512,
                "width": 512,
                "num_inference_steps": steps,
                "seed": int(fields.get("seed", "1500")),
                "include_reference_audio": False,
                "generate_audio": generate_audio,
                "return_video_b64": True,
            }
            set_status(job_id, state="generating", detail="Running H3 generation on Modal", progress=20)
            result = post_json(WARM_ENDPOINT_URL, payload)
            video_b64 = result.pop("video_b64", "")
            (job_dir / "modal.log").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if not video_b64:
                raise RuntimeError("warm endpoint did not return video data")
            set_status(job_id, state="downloading", detail="Receiving MP4", progress=90)
            local_out.write_bytes(base64.b64decode(video_b64))
        else:
            set_status(job_id, state="uploading", detail="Uploading reference to Modal", progress=10)
            run(["modal", "volume", "put", "--force", "h3-avatar-runs", str(ref), remote_ref])
            set_status(job_id, state="generating", detail="Running H3 generation on Modal", progress=20)
            cmd = [
                "modal", "run", "scripts/helpers/modal_h3_avatar.py::generate_ref2va_sample",
                "--checkpoint", checkpoint,
                "--reference-video", f"/runs{remote_ref}",
                "--output-path", container_out,
                "--prompt", prompt,
                "--num-frames", str(frames),
                "--height", "512",
                "--width", "512",
                "--num-inference-steps", str(steps),
                "--seed", str(int(fields.get("seed", "1500"))),
                "--no-include-reference-audio",
            ]
            cmd.append("--generate-audio" if generate_audio else "--no-generate-audio")
            modal_log = run(cmd)
            (job_dir / "modal.log").write_text(modal_log, encoding="utf-8")

            set_status(job_id, state="downloading", detail="Downloading MP4", progress=90)
            run(["modal", "volume", "get", "--force", "h3-avatar-runs", remote_out, str(job_dir) + "/"])
        downloaded = job_dir / "generated.mp4"
        if not downloaded.exists():
            matches = list(job_dir.glob("*generated*.mp4"))
            if matches:
                shutil.move(str(matches[0]), str(downloaded))
        set_status(job_id, state="done", detail="Done", progress=100, video_url=f"/jobs/{job_id}/generated.mp4")
    except Exception as exc:
        set_status(job_id, state="error", detail=str(exc), progress=100)


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H3 Avatar Studio</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1016; --panel:#171b23; --panel2:#11151d; --line:#2a3240; --text:#f6f8fc; --muted:#99a3b3; --accent:#6bc8ff; --accent2:#a8f0cf; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background:radial-gradient(circle at top left, #172233 0, #0d1016 380px); color:var(--text); }
    main { min-height:100vh; max-width:1240px; margin:0 auto; padding:36px; display:grid; grid-template-columns: 420px 1fr; gap:28px; align-items:stretch; }
    h1 { font-size:34px; line-height:1; margin:0 0 10px; letter-spacing:0; }
    p { color:var(--muted); line-height:1.5; }
    .intro { margin:0 0 24px; max-width:32rem; font-size:16px; }
    form, .result { background:rgba(23, 27, 35, .92); border:1px solid var(--line); border-radius:8px; box-shadow:0 24px 80px rgba(0,0,0,.28); }
    form { padding:22px; }
    .result { padding:22px; display:flex; flex-direction:column; min-height:620px; }
    label { display:block; margin:18px 0 8px; color:#d9e0eb; font-size:13px; font-weight:700; }
    textarea, button, input[type="text"] { width:100%; border-radius:7px; font:inherit; }
    textarea { min-height:112px; resize:vertical; border:1px solid var(--line); background:#0c1017; color:var(--text); padding:14px; line-height:1.45; }
    input[type="text"] { border:1px solid var(--line); background:#0c1017; color:var(--text); padding:12px; }
    textarea:focus, .upload:focus-within { outline:2px solid rgba(107,200,255,.42); outline-offset:2px; }
    button { margin-top:18px; background:linear-gradient(135deg, var(--accent), var(--accent2)); color:#061018; border:0; padding:14px 16px; font-weight:800; cursor:pointer; }
    button:disabled { opacity:.65; cursor:wait; }
    video { width:100%; max-height:72vh; background:#050608; border:1px solid var(--line); border-radius:8px; }
    .upload { display:grid; place-items:center; min-height:180px; padding:18px; border:1px dashed #405066; border-radius:8px; background:var(--panel2); cursor:pointer; text-align:center; }
    .upload input { position:absolute; inline-size:1px; block-size:1px; opacity:0; pointer-events:none; }
    .upload strong { display:block; color:#f8fbff; font-size:17px; margin-bottom:6px; }
    .upload span { display:block; color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
    .locked { margin:16px 0 0; padding:12px; border:1px solid #314458; border-radius:8px; background:#101822; color:#b7c7d9; font-size:13px; }
    .background-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .bg-card { border:1px solid var(--line); background:#0d121a; color:#dfe7f3; border-radius:8px; padding:12px; text-align:left; cursor:pointer; }
    .bg-card[aria-pressed="true"] { border-color:var(--accent); background:#112231; box-shadow:inset 0 0 0 1px rgba(107,200,255,.45); }
    .bg-card strong { display:block; font-size:13px; margin-bottom:4px; color:#f8fbff; }
    .bg-card span { display:block; color:var(--muted); font-size:12px; line-height:1.35; }
    .custom-bg { margin-top:10px; display:none; }
    .custom-bg.active { display:block; }
    .bar { height:10px; background:#0c0e13; border-radius:999px; overflow:hidden; border:1px solid var(--line); margin:18px 0 4px; }
    .fill { height:100%; width:0%; background:linear-gradient(90deg, var(--accent), var(--accent2)); transition:width .3s ease; }
    .pill { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:5px 10px; color:#cbd3df; font-size:12px; text-transform:capitalize; }
    .result-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }
    .result-title { margin:0; color:#f5f8fc; font-weight:800; }
    .empty { flex:1; display:grid; place-items:center; border:1px solid var(--line); border-radius:8px; background:#0a0d12; margin-top:18px; min-height:360px; color:#717c8d; text-align:center; padding:24px; }
    #job { font-size:12px; overflow-wrap:anywhere; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; padding:18px; } .result { min-height:420px; } h1 { font-size:30px; } }
  </style>
</head>
<body>
<main>
  <section>
    <h1>H3 Avatar Studio</h1>
    <p class="intro">Upload a character image, write what it should say, and generate a talking avatar clip with the locked H3 LoRA.</p>
    <form id="form">
      <label>Reference Image</label>
      <label class="upload">
        <input id="image" name="image" type="file" accept="image/*" required>
        <span>
          <strong>Choose a reference image</strong>
          <span id="fileName">PNG, JPG, or JPEG</span>
        </span>
      </label>
      <label>Dialogue</label>
      <textarea name="dialogue">Hey Chandan, wassup yaar, how are you doing these days?</textarea>
      <label>Direction</label>
      <textarea name="prompt">The same character faces the camera and speaks with an audible natural voice, expressive synchronized mouth movement, subtle head motion, and a friendly expression.</textarea>
      <label>Background</label>
      <input id="background" name="background" type="hidden" value="in a clean studio with soft cinematic lighting">
      <div class="background-grid" id="backgroundGrid">
        <button class="bg-card" type="button" aria-pressed="true" data-bg="in a clean studio with soft cinematic lighting">
          <strong>Studio</strong><span>Soft light, clean backdrop</span>
        </button>
        <button class="bg-card" type="button" aria-pressed="false" data-bg="inside a modern news studio with subtle screens in the background">
          <strong>Newsroom</strong><span>Modern broadcast desk feel</span>
        </button>
        <button class="bg-card" type="button" aria-pressed="false" data-bg="in a cozy room with warm lamps and shelves behind them">
          <strong>Cozy Room</strong><span>Warm indoor setting</span>
        </button>
        <button class="bg-card" type="button" aria-pressed="false" data-bg="on a city street at night with colorful lights behind them">
          <strong>City Night</strong><span>Neon evening backdrop</span>
        </button>
        <button class="bg-card" type="button" aria-pressed="false" data-bg="near a bright beach with blue sky and soft sunlight">
          <strong>Beach</strong><span>Bright outdoor scene</span>
        </button>
        <button class="bg-card" type="button" aria-pressed="false" data-bg="custom">
          <strong>Custom</strong><span>Type your own background</span>
        </button>
      </div>
      <div class="custom-bg" id="customBgWrap">
        <input id="customBg" type="text" placeholder="e.g. inside a spaceship control room">
      </div>
      <div class="locked">Locked settings: audio-v2 LoRA, model-generated audio, 10 second clip, 30 inference steps.</div>
      <button type="submit">Generate Video</button>
    </form>
  </section>
  <section class="result">
    <div class="result-head">
      <p class="result-title">Generation</p>
      <span class="pill" id="state">idle</span>
    </div>
    <p id="detail">Upload an image to start.</p>
    <div class="bar"><div class="fill" id="fill"></div></div>
    <p id="job"></p>
    <video id="video" controls playsinline hidden></video>
    <div class="empty" id="empty">Your generated clip will appear here.</div>
  </section>
</main>
<script>
const form = document.getElementById('form');
const imageInput = document.getElementById('image');
const fileName = document.getElementById('fileName');
const backgroundInput = document.getElementById('background');
const backgroundGrid = document.getElementById('backgroundGrid');
const customBgWrap = document.getElementById('customBgWrap');
const customBg = document.getElementById('customBg');
const state = document.getElementById('state');
const detail = document.getElementById('detail');
const fill = document.getElementById('fill');
const jobEl = document.getElementById('job');
const video = document.getElementById('video');
const empty = document.getElementById('empty');
let timer = null;
imageInput.addEventListener('change', () => {
  fileName.textContent = imageInput.files.length ? imageInput.files[0].name : 'PNG, JPG, or JPEG';
});
function setBackground(value) {
  if (value === 'custom') {
    customBgWrap.classList.add('active');
    backgroundInput.value = customBg.value.trim();
    customBg.focus();
  } else {
    customBgWrap.classList.remove('active');
    backgroundInput.value = value;
  }
}
backgroundGrid.addEventListener('click', (event) => {
  const card = event.target.closest('.bg-card');
  if (!card) return;
  document.querySelectorAll('.bg-card').forEach((item) => item.setAttribute('aria-pressed', 'false'));
  card.setAttribute('aria-pressed', 'true');
  setBackground(card.dataset.bg);
});
customBg.addEventListener('input', () => {
  const customSelected = document.querySelector('.bg-card[aria-pressed="true"]')?.dataset.bg === 'custom';
  if (customSelected) backgroundInput.value = customBg.value.trim();
});
function render(s) {
  state.textContent = s.state || 'unknown';
  detail.textContent = s.detail || '';
  fill.style.width = `${s.progress || 0}%`;
  if (s.job_id) jobEl.textContent = `job ${s.job_id}`;
  if (s.video_url) {
    video.hidden = false;
    empty.hidden = true;
    video.src = `${s.video_url}?t=${Date.now()}`;
  }
}
async function poll(id) {
  const r = await fetch(`/api/status?id=${encodeURIComponent(id)}`);
  const s = await r.json();
  render(s);
  if (s.state === 'done' || s.state === 'error') clearInterval(timer);
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  video.hidden = true;
  empty.hidden = false;
  const data = new FormData(form);
  render({state:'starting', detail:'Submitting job', progress:1});
  const r = await fetch('/api/generate', {method:'POST', body:data});
  const s = await r.json();
  render(s);
  clearInterval(timer);
  timer = setInterval(() => poll(s.job_id), 2500);
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/health":
            self.send_json({"ok": True})
            return
        if parsed.path == "/api/status":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            payload = STATUS.get(job_id)
            if payload is None:
                status_file = JOBS / job_id / "status.json"
                payload = json.loads(status_file.read_text()) if status_file.is_file() else {"state": "missing", "job_id": job_id}
            self.send_json(payload)
            return
        match = re.fullmatch(r"/jobs/([a-f0-9-]+)/generated\.mp4", parsed.path)
        if match:
            path = JOBS / match.group(1) / "generated.mp4"
            if not path.is_file():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        fields, files = parse_multipart(self.headers.get("Content-Type", ""), self.rfile.read(length))
        if "image" not in files:
            self.send_json({"state": "error", "detail": "missing image"}, 400)
            return
        job_id = str(uuid.uuid4())
        job_dir = JOBS / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        filename, content = files["image"]
        suffix = Path(filename).suffix.lower() or ".png"
        image_path = job_dir / f"input{suffix}"
        image_path.write_bytes(content)
        set_status(job_id, state="queued", detail="Queued", progress=0)
        thread = threading.Thread(target=generate_job, args=(job_id, image_path, fields), daemon=True)
        thread.start()
        self.send_json(STATUS[job_id])

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    port = int(os.environ.get("H3_UI_PORT", "8787"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"H3 Avatar Studio: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

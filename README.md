# RehabVLM

**Tiny vision-language models (3M parameters each)** for motor function and range-of-motion (ROM) assessment in rehabilitation:

* **Body model:** shoulders, elbows, hips, trunk lean, left/right asymmetry.
* **Hand model:** finger flexion (fist), pinch, finger spread, the most limited finger.

You show it a body pose or a hand, ask a question in plain language, and it answers with angles, grades and simple exercise suggestions.

It runs **fully in the browser** (ONNX Runtime Web + MediaPipe), so a patient's photo never leaves their phone, and the whole thing is hosted for free as a static site.

**Live demo:** https://rehabvlm.vercel.app (works on phones; switch between **Body** and **Hand**, try an example or take a photo)

```
photo ──► on-device pose estimation ──► colour-coded skeleton (96×96) ──► RehabVLM ──► "right shoulder abduction 60 degrees, limited ..."
                (MediaPipe)                (red = right, green = left)      (CNN-ViT → projector → decoder)
                                                       ▲
                                                question tokens
```

## Use it in your own app (API)

Get a free key at **https://rehabvlm.vercel.app/developers.html**. The page also gives you a ready-to-paste environment variable line:

```bash
# .env  (keep it out of git)
REHABVLM_API_KEY=rvlm_...
```

Then send 13 body joints, or 21 hand landmarks, from any tracker:

```bash
curl https://rehabvlm.vercel.app/api/v1/assess \
  -H "Authorization: Bearer $REHABVLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "how is my left shoulder?", "joints": {"nose": [0.30, 0.23], "l_shoulder": [0.40, 0.31], ...}}'

curl https://rehabvlm.vercel.app/api/v1/assess \
  -H "Authorization: Bearer $REHABVLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "make a fist", "hand": [[x, y, z], ... 21 landmarks in MediaPipe Hands order]}'
```

The response has the VLM's `answer`, plus the raw numbers (`angles_deg` for the body, `measurements` for the hand), which are handy for driving gameplay.

* **Unity / VR body:** drop [`RehabVLMClient.cs`](web/sdk/unity/RehabVLMClient.cs) into your project and assign the avatar's Humanoid Animator. It reads the 13 bones in the avatar's root space and calls the API.
* **Unity / VR hands:** add [`RehabVLMHandClient.cs`](web/sdk/unity/RehabVLMHandClient.cs) (needs `com.unity.xr.hands`). It reads the 21 tracked hand joints from OpenXR hand tracking (Quest, Pico, Vision Pro, Ultraleap). For a webcam hand tracker, call `AssessLandmarks()` instead.
* **Unity API key:** the clients read `REHABVLM_API_KEY` from the environment, or from a git-ignored `Assets/Resources/rehabvlm_env.txt` containing the same `REHABVLM_API_KEY=...` line (works on Quest/Android/iOS), or from the Inspector field.
* **Web / mobile / Python:** send MediaPipe's 33 pose landmarks as `"mediapipe": [[x, y], ...]`, or its 21 hand `worldLandmarks` as `"hand": [[x, y, z], ...]`.
* Keys are HMAC-signed, so no database is needed. Revoke a key by adding its id to the `RVLM_REVOKED_KEYS` env var. Rate limit: 60 requests/min per key (best effort, per server instance).

---

## Why a VLM and not an LLM?

**An LLM can only read text. Motor assessment starts from what the body looks like.**

1. **The evidence is visual.** Range of motion, posture, trunk lean and left/right asymmetry are things you *see*. To use an LLM, someone first has to measure the angles with a goniometer and type them in. That measuring step is exactly what we want to automate. A VLM takes the image as input and does the measuring itself.
2. **Grounded, not self-reported.** An LLM can only work from what the patient says ("I can't lift my arm very high"), which is subjective and easy to hallucinate on. A VLM's answer is conditioned on pixels, so "60 degrees" comes from the actual pose in front of it.
3. **One model, many questions.** A plain vision model (classifier or regressor) has a fixed output, for example 6 numbers. A VLM takes a *question* along with the image and generates a *natural-language* answer: "assess the shoulders", "which side is weaker?", "suggest exercises". New assessment questions are new training text, not a new architecture. The answer is also readable by patients and clinicians directly.
4. **It scales to what rules can't easily capture.** For a single joint angle, geometry on keypoints is enough (the demo shows that check side by side). The point of the VLM design is the path forward: compensation patterns (shoulder hiking, trunk lean while reaching), movement quality across video frames, and assistive-device use. These are visual patterns that are hard to hand-code but learnable from images plus text.
5. **Small enough for the edge.** At 3M parameters (about 12 MB), it runs on a phone in the browser. That makes it private and free to host, and it works in a clinic, a home or a low-resource setting.

**One-line version:** *an LLM needs someone to describe the patient; a VLM looks at the patient.*

---

## What it can answer

| Question | Example answer |
|---|---|
| describe the posture | trunk upright. right shoulder abduction 60 degrees. left shoulder abduction 165 degrees. right elbow flexion 85 degrees ... |
| assess the shoulders | right shoulder abduction 60 degrees, limited. left shoulder abduction 165 degrees, full range. asymmetry 105 degrees, significant. |
| assess the elbows / hips | per-side angle + grade |
| check symmetry | upper and lower limb asymmetry + trunk lean |
| which side is weaker? | the right side shows less range, 105 degrees less shoulder abduction. |
| suggest exercises | right shoulder is limited. try pendulum swings, wall finger walks and assisted arm raises. |

**Hand model:**

| Question | Example answer |
|---|---|
| describe the hand | thumb flexion 85 degrees. index flexion 250 degrees. ... ring flexion 45 degrees. ... fingers are flexed, spread not assessed. |
| assess the fist | average finger flexion 200 degrees, full fist. least flexed is the ring finger at 45 degrees. |
| assess the pinch | thumb to index gap 20 percent of hand width, pinch achieved. thumb flexion 40 degrees. |
| which finger is most limited? | the ring finger has the least flexion, 45 degrees. |
| check finger spread | finger spread 35 degrees, moderate. |
| suggest hand exercises | the ring finger lags behind. try isolated ring finger bends and blocking exercises. |

Finger flexion is summed over a finger's joints (MCP + PIP + DIP; thumb MCP + IP), so a full fist is about 250°. Fist grades (average of the four fingers): ≥200° full fist, 120–200° partial fist, 40–120° slightly flexed, <40° open hand. Pinch: thumb-to-index gap <25% of hand width = achieved, <60% = close.

Free-text questions in the demo are routed to the closest supported question.

Grading thresholds (shoulder abduction during a maximal arm raise): ≥150° full range, 90–150° partial, <90° limited. Asymmetry: <15° minimal, 15–40° moderate, >40° significant.

## Architecture

LLaVA-style prefix-LM, trained from scratch:

| Part | Details |
|---|---|
| Vision encoder | CNN stem with CoordConv (x/y coordinate channels) → 12×12 grid = 144 image tokens, 4-layer ViT, d=160 |
| Projector | 2-layer MLP into the language space |
| Language decoder | 4-layer transformer, closed word vocabulary (118 words body / 140 hand; numbers are tokens in 5° steps) |
| Attention | image tokens attend to each other bidirectionally; text tokens attend causally to image + previous text |
| Training loss | next-token cross-entropy on the answer + auxiliary joint-angle regression on the image features |
| Size | 2.96M parameters per model, about 12 MB as ONNX. Body and hand are separate models with the same architecture |

### What we learned building it

The first version used a plain patch-ViT encoder. It learned fluent answers but **ignored the image**: angle errors were at chance level (~59°), because the decoder fell back on a language-only prior. Two changes fixed this:

* **CoordConv + CNN stem.** Feeding x/y coordinate maps lets the encoder localise thin skeleton lines. Angle regression error fell from ~30° to ~8° within 300 steps.
* **Auxiliary angle loss.** A small head regresses the 7 joint angles from the image tokens, which forces the encoder to learn geometry that the decoder can then put into words.

## Data

Training data is **procedurally generated**, so it can be made without patient data and without a GPU:

* Frontal-plane poses are sampled from clinically motivated profiles: healthy, right/left hemiparesis, right/left frozen shoulder, resting stance, and random. Body proportions vary, and landmark jitter imitates a real pose detector.
* Each pose is drawn as a colour-coded skeleton (subject's right = red, left = green, trunk/head = blue) with a random line width.
* Ground-truth angles are *measured* from the keypoints. The question/answer pairs are built from templates.

Real photos go through MediaPipe Pose on the device, and the landmarks are drawn with **the same rasteriser** (`rehabvlm/pose.py` ⇄ `web/core.mjs`), so the model sees real and synthetic inputs in the same format. A side benefit is privacy: the model never sees a face or the surroundings.

**Hands** (`rehabvlm/hand.py`) are generated in 3D from a kinematic hand model with clinically motivated postures: open hand, full fist, partial fist, pinch, claw hand, spastic (stroke) hand, a single stiff finger, and random. Each hand is then randomly rotated, scaled, mirrored (left/right) and jittered like a tracker. Before the model sees it, the hand is put into a **canonical frame** (wrist at the origin, middle knuckle up, index to one side) and drawn from **three orthographic views**, one per image channel: front, side and top. The side and top views make finger curl visible, which a single 2-D view hides. The result is independent of how the hand is held, so the same model works for webcams (MediaPipe `worldLandmarks`) and VR hand tracking.

## Results

Evaluated on 300 fresh synthetic poses (`python evaluate.py`):

| Joint angle read by the VLM | Mean absolute error |
|---|---|
| Shoulder abduction (R / L) | 1.8° / 1.7° |
| Elbow flexion (R / L) | 2.5° / 2.3° |
| Hip abduction (R / L) | 1.4° / 1.5° |
| **Overall** | **1.9°** |

For comparison, the first version (plain ViT, no auxiliary loss) scored about 59° on shoulders, which is chance level.

Exact match with the full reference sentence (every angle in the right 5° bin and every grade correct): weaker side 72%, hips 69%, elbows 46%, shoulders 45%, symmetry 44%, exercise suggestion 98%, full posture description 21% (seven numbers must all land in the same 5° bin). Most misses are one bin off.

**Hand model** (300 fresh synthetic hands, random orientation, mirroring and tracker noise):

| Finger flexion read by the VLM (summed over the finger's joints) | Mean absolute error |
|---|---|
| Thumb / index / middle / ring / pinky | 5.1° / 9.1° / 9.5° / 8.1° / 9.0° |
| **Overall** (on a 0–270° scale) | **8.2°** |

| Clinical call | Accuracy |
|---|---|
| Fist grade (full / partial / slightly flexed / open) | 95.0% |
| Pinch (achieved / close / not achieved) | 90.5% |
| Finger spread grade | 96.0% |
| Exercise advice category | 90.0% |
| Most limited finger, when it lags >20° behind the next one | 93.2% (72.5% including near-ties) |

> These numbers are on synthetic skeletons. Accuracy on real photos also depends on the pose estimator and camera angle.

## Run it yourself

```bash
pip install -r requirements.txt
python train.py --steps 4000          # body model, ~2 h on a 4-core CPU (trained weights are in checkpoints/)
python train.py --domain hand --steps 4000 --workers 2 --out checkpoints/rehabvlm-hand.pt
python evaluate.py --n 300
python evaluate.py --ckpt checkpoints/rehabvlm-hand.pt --n 300
python export_onnx.py                                         # -> web/model/
python export_onnx.py --ckpt checkpoints/rehabvlm-hand.pt     # -> web/model/hand/
cd web && python -m http.server 8000  # open http://localhost:8000
```

## Deploy for free

`web/` is a static site with no server and no GPU.

* **Vercel:** import this repo; `vercel.json` already points the output directory to `web/`.
* **GitHub Pages:** Settings → Pages → deploy from branch, folder `/web` (or copy `web/` to `docs/`).
* **Hugging Face Spaces:** create a *Static* Space and upload the contents of `web/`.

## Limitations and next steps

* Trained only on synthetic skeletons, so accuracy on real photos depends on the pose estimator and on camera angle. The body model is frontal view only: angles are measured in the image plane.
* Hand: webcam depth (z) from MediaPipe is noisier than VR hand tracking, so expect better finger readings from Quest/OpenXR hands. MCP hyperextension is reported as flexion. Wrist movement is not covered, because the 21 hand landmarks have no forearm point.
* Closed vocabulary. It answers the supported question types and does not hold open conversation.
* Next steps: fine-tune on real clinical images/video (for example, with goniometer-labelled ROM), add sagittal-view tasks (knee/hip flexion, gait), feed raw image crops alongside the skeleton, and add temporal tokens for movement quality across video frames.

> **Disclaimer:** research and education project, not a medical device. Do not use it for diagnosis or treatment decisions.

## Project layout

```
rehabvlm/pose.py    synthetic poses, angle measurement, skeleton rasteriser
rehabvlm/hand.py    synthetic 3-D hands, finger measurements, 3-view rasteriser
rehabvlm/text.py    body questions, answer templates, tokenizer
rehabvlm/hand_text.py  hand questions and answer templates
rehabvlm/domains.py registry so training/eval/export work for "body" and "hand"
rehabvlm/model.py   TinyVLM (ViT encoder + projector + decoder)
train.py            training on an endless synthetic stream
evaluate.py         angle error + exact-match evaluation
export_onnx.py      ONNX export + demo poses for the web app
web/                static site: demo (index.html, app.js), developer page, model/, sdk/unity/
web/core.mjs        body + hand rasterisers, measurements and VLM decoding, shared by the browser and the API
api/                serverless API: /api/keys, /api/v1/assess, /api/v1/health
```

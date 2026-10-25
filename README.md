# RehabVLM

A **tiny vision-language model (3M parameters)** for motor function and range-of-motion (ROM) assessment in rehabilitation. You show it a body pose, ask a question in plain language, and it answers with joint angles, grades, left/right asymmetry and simple exercise suggestions.

It runs **fully in the browser** (ONNX Runtime Web + MediaPipe), so a patient's photo never leaves their phone, and the whole thing is hosted for free as a static site.

**Live demo:** _see the repository's website link_

```
photo ──► on-device pose estimation ──► colour-coded skeleton (96×96) ──► RehabVLM ──► "right shoulder abduction 60 degrees, limited ..."
                (MediaPipe)                (red = right, green = left)      (CNN-ViT → projector → decoder)
                                                       ▲
                                                question tokens
```

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

Free-text questions in the demo are routed to the closest supported question.

Grading thresholds (shoulder abduction during a maximal arm raise): ≥150° full range, 90–150° partial, <90° limited. Asymmetry: <15° minimal, 15–40° moderate, >40° significant.

## Architecture

LLaVA-style prefix-LM, trained from scratch:

| Part | Details |
|---|---|
| Vision encoder | CNN stem with CoordConv (x/y coordinate channels) → 12×12 grid = 144 image tokens, 4-layer ViT, d=160 |
| Projector | 2-layer MLP into the language space |
| Language decoder | 4-layer transformer, 118-word vocabulary (angles are tokens in 5° steps) |
| Attention | image tokens attend to each other bidirectionally; text tokens attend causally to image + previous text |
| Training loss | next-token cross-entropy on the answer + auxiliary joint-angle regression on the image features |
| Size | 2.96M parameters, about 12 MB as ONNX |

### What we learned building it

The first version used a plain patch-ViT encoder. It learned fluent answers but **ignored the image**: angle errors were at chance level (~59°), because the decoder fell back on a language-only prior. Two changes fixed this:

* **CoordConv + CNN stem.** Feeding x/y coordinate maps lets the encoder localise thin skeleton lines. Angle regression error fell from ~30° to ~8° within 300 steps.
* **Auxiliary angle loss.** A small head regresses the 7 joint angles from the image tokens, which forces the encoder to learn geometry that the decoder can then put into words.

## Data

Training data is **procedurally generated**, so it can be made without patient data and without a GPU:

* Frontal-plane poses are sampled from clinically motivated profiles: healthy, right/left hemiparesis, right/left frozen shoulder, resting stance, and random. Body proportions vary, and landmark jitter imitates a real pose detector.
* Each pose is drawn as a colour-coded skeleton (subject's right = red, left = green, trunk/head = blue) with a random line width.
* Ground-truth angles are *measured* from the keypoints. The question/answer pairs are built from templates.

Real photos go through MediaPipe Pose on the device, and the landmarks are drawn with **the same rasteriser** (`rehabvlm/pose.py` ⇄ `web/app.js`), so the model sees real and synthetic inputs in the same format. A side benefit is privacy: the model never sees a face or the surroundings.

## Results

Evaluated on 300 fresh synthetic poses (`python evaluate.py`):

RESULTS_PLACEHOLDER

## Run it yourself

```bash
pip install -r requirements.txt
python train.py --steps 4000          # ~2 h on a 4-core CPU
python evaluate.py --n 300
python export_onnx.py                 # writes web/model/{encoder,decoder}.onnx + vocab
cd web && python -m http.server 8000  # open http://localhost:8000
```

## Deploy for free

`web/` is a static site with no server and no GPU.

* **Vercel:** import this repo; `vercel.json` already points the output directory to `web/`.
* **GitHub Pages:** Settings → Pages → deploy from branch, folder `/web` (or copy `web/` to `docs/`).
* **Hugging Face Spaces:** create a *Static* Space and upload the contents of `web/`.

## Limitations and next steps

* Trained only on synthetic skeletons, so accuracy on real photos depends on the pose estimator and on camera angle. Frontal view only: angles are measured in the image plane.
* Closed vocabulary. It answers the supported question types and does not hold open conversation.
* Next steps: fine-tune on real clinical images/video (for example, with goniometer-labelled ROM), add sagittal-view tasks (knee/hip flexion, gait), feed raw image crops alongside the skeleton, and add temporal tokens for movement quality across video frames.

> **Disclaimer:** research and education project, not a medical device. Do not use it for diagnosis or treatment decisions.

## Project layout

```
rehabvlm/pose.py    synthetic poses, angle measurement, skeleton rasteriser
rehabvlm/text.py    questions, answer templates, tokenizer
rehabvlm/model.py   TinyVLM (ViT encoder + projector + decoder)
train.py            training on an endless synthetic stream
evaluate.py         angle error + exact-match evaluation
export_onnx.py      ONNX export + demo poses for the web app
web/                static browser app (index.html, app.js, model/)
```

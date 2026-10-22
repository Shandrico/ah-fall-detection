# Jetson transfer quick start

The source branch is `lifan-latesst`. Keep recordings, participant metadata,
real calibrations, transfer manifests and model weights on approved storage,
not GitHub. No hospital footage is needed in the repository.

## Move the SSD, not another copy of the footage

1. Confirm the latest code was pushed and the SSD handoff file checks passed.
2. Safely eject the T9 from Windows, then connect it to a Jetson USB 3 port.
   Do not format it if Linux cannot mount it; resolve the mounting problem first.
3. Open a terminal in the SSD's `AHFD_JETSON_HANDOFF_...` folder and run:

   ```bash
   python3 jetson_preflight.py
   ```

   The checker reads the external `transfer_manifest.json`, checks the small
   handoff artifacts by SHA-256, and checks recording sizes and populated
   colour/depth topics. It also reports board/L4T/Python and package/provider
   availability. It installs nothing, starts no camera and does not train.
   Raw file size checks are not full-file integrity verification. Missing GPU
   dependencies are reported separately from transfer-file success.

From a Git checkout, the same check is available as:

```bash
python3 scripts/jetson_preflight.py --kit /actual/ssd/mount/AHFD_JETSON_HANDOFF_...
```

Use `--files-only` for file checks without importing optional ML packages.
Paths above are placeholders; use the drive's actual Linux mount location.

## Restore the code

With network access, clone onto the Jetson's internal Linux filesystem:

```bash
git clone --branch lifan-latesst https://github.com/Shandrico/ah-fall-detection.git
cd ah-fall-detection
git rev-parse HEAD
```

Compare the revision with `source_commit` in the SSD manifest. A later remote
revision is not the exact verified handoff revision; choose deliberately.

An accompanying Git bundle provides an offline alternative:

```bash
git clone --branch lifan-latesst /actual/path/to/ah-fall-detection-REV.bundle
```

Use the actual bundle filename from the handoff folder. The source ZIP is also
usable for inspection/replay, but is not a Git checkout. The governed collector
requires a clean checkout and its normal approval/calibration checks.

Keep the Windows project until the Jetson can open the data and run the code.
Before deleting any unique local work, confirm it is tracked/pushed or backed
up separately. Git does not preserve ignored recordings, environments or weights.
Do not treat a sole portable SSD as a backup of itself.

## Runtime setup is a separate gate

- Verify the actual Jetson model and JetPack/L4T release. This project targets
  Orin NX 16 GB, but the checker must read the actual board.
- This source requires Python `>=3.10,<3.11`. Create a Linux environment rather
  than copying a Windows venv. Do not replace the system Python or flash JetPack
  automatically to resolve a version mismatch.
- The laptop's `openvino/gpu` setting is for Intel hardware. The supplied
  [replay smoke profile](../configs/jetson_replay_smoke.yaml) requests
  `onnxruntime/cuda`, uses skeleton display and disables detection/alerts/raw
  recording. Replace its source placeholder before use. It is not an approved
  collection or clinical configuration.
- Install a Linux/aarch64 ONNX Runtime GPU build compatible with the board's
  CUDA/cuDNN stack. Do not blindly install dependencies over a prepared GPU
  environment: `pyproject.toml` names `onnxruntime`, the CPU package. Resolve
  compatible dependencies first, then install project code with `--no-deps`.
- Provider availability is not a successful inference test. Verify the model
  actually uses CUDA and can replay the recorded `.db3` format, including zstd.
  The RTMO wrapper does not currently offer a direct TensorRT backend option.
- Put the supplied pretrained RTMO-S ONNX file into the account's rtmlib
  `hub/checkpoints` cache with its original basename if offline operation is
  needed. Check existing files before replacing them. A compiled TensorRT engine
  is not a portable substitute for these ONNX weights.

References: [NVIDIA Jetson documentation](https://docs.nvidia.com/jetson/index.html),
[ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Training after transfer

First review actual labels and per-camera geometry/alignment, then stream one
recording at a time into compact features. The small logistic/tree classifiers
can train on CPU; the Jetson GPU is useful for fixed pose inference/extraction.
Do not retrain RTMO for this short pilot.

For the ten-take pilot, use takes 01-08 for development and keep 09-10 held out.
Both camera views of a take stay in one partition; adjacent take pairs also stay
together as A-D development blocks. Confirm the filename-to-take mapping rather
than assuming it is ground truth. Ten takes viewed by two cameras are not twenty
independent trials. Keep the older dataset in a separate experiment.

The one-minute raw recordings are not the completed derived-only sessions
expected by `compare-bed-exit`. They still need reviewed labels, extraction and
a fixed-holdout evaluation path. The original `collection-plan` continues to
support 40/60-minute plans; this handoff does not change that protocol or start
training. Transfer success is not deployment or clinical validation.

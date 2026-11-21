# Evaluating red-light crossings

The unit of evaluation is one vehicle crossing a directed stop line during red. On eight generated clips, the motion pipeline recovered **five of six crossings** and emitted **zero unmatched candidates**. An overhead obstruction hid the sixth crossing. Every reported plate matched the generator label on the five visible events. Field accuracy, including field OCR accuracy, remains unmeasured.

## Reproduce the experiment

Install the project and the English Tesseract runtime using the [setup guide](../README.md). Run from the project root.

```sh
python -m trafficlab.experiment --out runs/reference
```

The command generates videos and annotations, then decodes the saved Motion JPEG files and runs the same analyzer used by the CLI. Ground truth comes from vehicle trajectories and signal phases before rasterization. The analyzer receives video pixels, camera calibration and the recording start time. Its event callback captures the predictions for scoring.

```sh
# Generate input without inference, including an optional README preview
python -m trafficlab.experiment --out runs/lab --generate-only --preview runs/lab/daylight.gif

# Rerun the analyzer against the exact saved bytes and calibration
python -m trafficlab.experiment --out runs/lab --evaluate-existing

# Ablate OCR. A fresh run overwrites report.json inside the chosen directory.
python -m trafficlab.experiment --out runs/lab --evaluate-existing --no-ocr
```

The run records dataset and code fingerprints, library versions, per-clip outcomes and elapsed time. A seed fixes image noise and the trajectories are deterministic. Encoder versions may change the video bytes. Use the stored input hashes when comparing two runs on identical media. [Reference manifest](research/manifest.json) · [Reference predictions, matches and environment](research/reference.json)

## Reference run

Each clip contains 144 frames at 12 fps and 768 × 432 pixels. The complete synthetic dataset covers 96 seconds and uses seed 4815. It contains two lane positions, two rendered vehicle colors, two invented plate strings, a single lens geometry and a single font. All clips share the same intersection. These are development fixtures, including a deliberately unobservable case.

| Scene | Physical red crossings | Matched | Unmatched candidates | Condition |
| --- | ---: | ---: | ---: | --- |
| daylight | 1 | 1 | 0 | A second car crosses after the signal turns green |
| parallel | 2 | 2 | 0 | Two vehicles cross in the same frame in separate lanes |
| night | 1 | 1 | 0 | Reduced brightness and deterministic pixel noise |
| occlusion | 1 | 1 | 0 | A fixed object partly obscures the vehicle on approach |
| stop-reverse | 0 | 0 | 0 | One vehicle stops upstream and another crosses in reverse |
| ambiguous | 0 | 0 | 0 | Simultaneous red and green indications |
| amber | 0 | 0 | 0 | Vehicle crosses on amber |
| hidden-crossing | 1 | 0 | 0 | An overhead structure completely hides the vehicle at the line |

**Event precision** was 5/5, **event recall** was 5/6 and **event F1** was 10/11 (0.909). **Plate exact match** was 5/5 among correctly detected, labeled events. The pipeline recovered the crossing and its plate together on 5/6 physical events. The three control clips and the legal crossing in the daylight clip produced no candidates.

The hidden crossing counts as a miss because the simulator knows the vehicle trajectory. No frame shows that vehicle passing the stop line. The reviewer receives no case for this event. The distinction matters when deciding where to mount a camera and how to label visibility in a field study.

The reference timing covers video decode, detector initialization per clip, tracking, signal classification, event OCR and JPEG evidence generation on the developer machine. It excludes fixture generation, annotation checks, scoring, database writes and HTTP traffic. The JSON records the wall time and aggregate processed frames per second across all 1,152 input frames. This is a single run of the CPU motion backend. Throughput depends on CPU, image size, event density, OCR and concurrency.

## Matching and denominators

1. Match events within each clip. A valid pair has an absolute timing error of at most four frames and Euclidean distance at most 0.07 between normalized image positions. A position uses the center of the vehicle box’s lower edge. At the fixture frame rate, the timing window is one third of a second.
2. Use linear sum assignment to maximize the number of valid pairs. Break ties by minimizing the sum of normalized temporal and spatial errors. A double detection creates an unmatched prediction. Plate text and the analyzer’s track ID have no role in matching.
3. Sum counts across all clips. Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2TP/(2TP+FP+FN). An undefined ratio is JSON `null`.
4. Compare plate strings exactly for matched events with a supplied reference plate. An abstention counts as a failed read. The report separately records how often a plate was returned, how often it was correct, and whether OCR was enabled.

## Authorized field evaluation

Use one directory per camera calibration. Supply `camera.json` and a version 1 `manifest.json` using the generated manifest as a concrete example. Set `generated` to `false` for camera footage and document collection, consent or permissions, annotation procedure, time span and camera conditions in `provenance`. Keep identifying media and plate annotations in access-controlled storage.

Each scene needs a unique name, a relative video path in `file`, its SHA-256 digest, decoded frame count, a timezone-aware ISO 8601 `recording_start`, and an array of `expected_events`. Each event specifies a zero-based frame and normalized x/y bottom center. Supply the adjudicated plate or `null` when unreadable. Optional per-frame labels require a path and SHA-256 digest. The runner checks file hashes, frame indexes, decode count and camera checksum. Edit and review labels before sealing the manifest.

```sh
python -m trafficlab.experiment --out private/held-out-camera --evaluate-existing --detector coco
```

Install the COCO extra and provision weights before this command. The reference result above covers the motion backend only. The trained detector’s ability to recognize real vehicles needs its own evaluation.

A separate integration check ran the official COCO weights on CPU with Torch 2.6.0 and TorchVision 0.21.0. On the left image from scikit-image’s stereo motorcycle pair, the backend returned one motorcycle. This verifies the model, color conversion and class filter on one photograph. It supplies no estimate of detection accuracy. [Image description](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle) · [Model and weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)

| Research question | Study design |
| --- | --- |
| Can the system generalize to a new intersection? | Split by camera and recording day. Freeze calibration and thresholds on development clips before testing. Hold out whole clips, including uneventful footage. |
| Which conditions create errors? | Label illumination, rain, glare, occlusion, class, crowded lanes and signal visibility. Report counts, missed crossings and false candidates per camera-hour for each group. |
| Can annotators agree on crossing time and plate text? | Have two reviewers independently label a subset, including uncertain events. Record disagreements, visibility, adjudication and inter-reviewer timing differences. |
| Where do errors originate? | Extend evaluation with vehicle AP, per-frame signal confusion, tracking IDF1/HOTA and OCR character error rate. Report component scores beside end-to-end event results. The generated frame labels include object IDs, clipped boxes and lamp states for such extensions. |
| How uncertain are the estimated rates? | Collect multiple sites and days and report confidence intervals clustered by camera or clip. Five visible synthetic positives provide too little independent evidence for a field claim. |

External benchmark tasks can support component studies if licenses and viewpoints fit the experiment. AI City publishes track-specific access terms, tasks and evaluation protocols. KITTI provides a car-tracking benchmark and documents its use of HOTA. A dedicated intersection dataset still needs synchronized signal, stop-line and event labels. [AI City data and evaluation](https://www.aicitychallenge.org/2024-data-and-evaluation/) · [KITTI tracking evaluation](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) · [TrackEval](https://github.com/JonathonLuiten/TrackEval)

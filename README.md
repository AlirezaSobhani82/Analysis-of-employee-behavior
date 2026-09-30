# Employee Activity Recognition and Productivity Analysis

## 1. Project Overview

This project is a Computer Vision pipeline for analyzing human activity in video and generating activity and productivity information for tracked people.

The system processes video frames, detects and tracks people, creates short person sequences, recognizes human actions, detects nearby objects, estimates body pose, combines the available information, applies decision rules, performs temporal smoothing, calculates productivity statistics, and generates a visualization.

The main objective is to determine the activity state of each tracked person as:

```text
WORKING
NOT_WORKING
NEUTRAL
```

---

## 2. Project Goal

The system is designed to answer the following questions:

1. Where are the people in the video?
2. Which frames belong to the same person?
3. What action is the person performing?
4. What objects are near or associated with the person?
5. What is the person's body pose?
6. How should the combined information be classified?
7. What is the person's working ratio and productivity according to the system?

---

## 3. Overall Pipeline

```text
Video
  |
  v
YOLO11 Person Detection
  |
  v
BoT-SORT Tracking
  |
  v
Stable Person ID
  |
  v
10-Frame Person Sequence
  |
  +-------------------------+
  |                         |
  v                         v
SlowFast Action         Object Detection
Recognition                  |
  |                          v
  |                    Object-Person
  |                     Association
  |                          |
  +------------+-------------+
               |
               v
        MediaPipe Pose
               |
               v
             Fusion
               |
               v
          Rule Engine
               |
               v
      Temporal Smoothing
               |
               v
 WORKING / NOT_WORKING / NEUTRAL
               |
               v
     Productivity Report
               |
               v
        Visualization
```

---

# 4. Technologies

The project uses:

- Python
- PyTorch
- PyTorchVideo
- OpenCV
- Ultralytics YOLO
- YOLO11
- BoT-SORT
- MediaPipe Pose
- NumPy
- Pandas
- SciPy
- PyYAML
- tqdm

---

# 5. Project Structure

```text
D:\SlowFast_Project
|
+-- data
|   +-- videos
|   |   +-- s_000.mp4
|   |   +-- s_001.mp4
|   |   +-- ...
|   |   +-- s_039.mp4
|   |
|   +-- labels
|       +-- s_000.txt
|       +-- s_001.txt
|       +-- ...
|       +-- s_039.txt
|
+-- models
|   +-- yolo11n.pt
|   +-- pose_landmarker_full.task
|   +-- best_model.pth
|
+-- outputs
|   +-- tracking
|   +-- action
|   +-- objects
|   +-- pose
|   +-- fusion
|   +-- decision
|   +-- productivity
|   +-- visualization
|   +-- evaluation
|
+-- train model slowfast
|   +-- evaluate.py
|
+-- README.md
+-- requirements.txt
```

---

# 6. Input Videos

The project contains 40 videos:

```text
s_000.mp4
s_001.mp4
...
s_039.mp4
```

Video directory:

```text
D:\SlowFast_Project\data\videos
```

---

# 7. Action Labels

The action annotation files are stored in:

```text
D:\SlowFast_Project\data\labels
```

The project contains:

```text
s_000.txt
s_001.txt
...
s_039.txt
```

Each annotation line follows this format:

```text
Class_ID,Start_Frame,End_Frame
```

Example:

```text
10,21,153
17,162,392
```

The labels represent temporal action segments.

They do not contain a Person ID.

Therefore, these labels are used for action-recognition evaluation and are not independent person-level Ground Truth for the final Working / Not Working decision.

---

# 8. Person Detection

YOLO11 is used to detect people in the video.

The system filters detections to the person class and obtains bounding boxes for detected people.

The detection stage provides the input required by the tracking stage.

Main model:

```text
models/yolo11n.pt
```

---

# 9. Person Tracking

BoT-SORT is used to track people across video frames.

The purpose of tracking is to maintain an identity for a person across consecutive frames.

Example:

```text
Frame 100 -> Person 1
Frame 101 -> Person 1
Frame 102 -> Person 1
Frame 103 -> Person 1
```

Tracking outputs are stored under:

```text
outputs/tracking
```

---

# 10. Stable Person ID

The project contains an additional stable-ID mechanism to reduce identity changes.

The stable identity process uses information such as:

- Previous bounding box
- Motion prediction
- Distance between current and previous positions
- Bounding-box smoothing
- Missing-frame tolerance

The result is a stable person identity used by later stages.

Example:

```text
person_1
person_2
person_3
```

---

# 11. Person Sequence

The current project uses:

```text
10 frames
```

for each person sequence used by the action-recognition stage.

The sequence contains consecutive frames belonging to the same tracked person.

This sequence is provided to the SlowFast model for temporal action recognition.

---

# 12. SlowFast Action Recognition

SlowFast is used to recognize human actions from the person sequences.

The project contains 18 action classes.

| Class ID | Action |
|---:|---|
| 1 | Typing |
| 2 | Writing |
| 3 | Drinking |
| 4 | Eating |
| 5 | Talking_in_phone |
| 6 | Talking_with_human |
| 7 | Shaking_hands |
| 8 | Thinking |
| 9 | Sleeping_on_desk |
| 10 | Close_Door |
| 11 | Open_Door |
| 12 | Sitting_down |
| 13 | Standing_up |
| 14 | Standing |
| 15 | Falling_down |
| 16 | Surprised |
| 17 | Walking |
| 18 | Sneezing |

---

# 13. SlowFast Model

The trained model used for evaluation is:

```text
D:\SlowFast_Project\models\best_model.pth
```

The evaluation script is:

```text
D:\SlowFast_Project\train model slowfast\evaluate.py
```

---

# 14. SlowFast Evaluation Results

The evaluated SlowFast model produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | **77.88%** |
| Precision | **75.22%** |
| Recall | **76.30%** |
| F1 Score | **74.08%** |

## Accuracy

```text
SlowFast Action Accuracy = 77.88%
```

This means that approximately 78 out of every 100 evaluated action samples were classified correctly by the SlowFast model.

## Important

The value:

```text
77.88%
```

is the **Action Recognition Accuracy of the SlowFast model**.

It is not the Accuracy of the complete Working / Not Working system.

---

# 15. Confusion Matrix

The evaluation produced the following confusion matrix:

```text
[[ 8  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  7  2  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0]
 [ 0  0  4  0  1  0  0  0  0  0  0  0  1  0  0  0  0  0]
 [ 0  1  5  5  0  0  0  0  0  0  0  0  0  0  0  0  0  1]
 [ 0  2  0  1  4  0  0  1  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  6  0  0  0  0  0  1  0  1  0  0  0  0]
 [ 0  0  0  0  0  0 11  0  0  0  0  0  0  0  1  0  0  0]
 [ 0  1  0  0  1  0  0  3  0  0  0  0  0  0  0  0  0  1]
 [ 0  1  0  0  0  0  0  0  7  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0 10  1  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  7  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  1  0  0  0  0  0 20  0  0  0  0  0  1]
 [ 0  0  0  0  0  0  1  0  0  0  0  1 21  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  4  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  1  0  0  0  1  0  0  0  0  0  1  0  4  0  1]
 [ 0  0  0  0  0  0  0  0  1  0  1  0  0  0  0  0 26  0]
 [ 0  1  3  3  1  0  0  1  0  0  0  0  0  2  0  0  0  5]]
```

The confusion matrix shows that some action classes are confused with other visually or temporally similar actions.

---

# 16. Object Detection

Object detection is used to identify objects that provide additional context about the person's activity.

Relevant objects include examples such as:

- Laptop
- Keyboard
- Mouse
- Cell phone
- Book
- Bottle
- Cup

Detected objects are associated with nearby tracked people.

Object context output:

```text
outputs/objects/object_context.csv
```

---

# 17. Pose Estimation

MediaPipe Pose is used to estimate body-pose information.

Pose information provides additional context for the activity decision.

Pose output:

```text
outputs/pose/pose_context.csv
```

---

# 18. Feature Fusion

The system combines information from:

- SlowFast Action Recognition
- Object Detection
- Pose Estimation

The fusion stage creates a combined representation for each tracked person and sequence.

Output:

```text
outputs/fusion/fusion.csv
```

---

# 19. Rule Engine

The Rule Engine uses the available action, object, and pose information to produce an activity state.

Possible states:

```text
WORKING
NOT_WORKING
NEUTRAL
```

## Working Actions

```text
Typing
Writing
Talking_in_phone
Talking_with_human
Shaking_hands
```

## Not Working Actions

```text
Drinking
Eating
Sleeping_on_desk
Walking
Sneezing
Falling_down
Surprised
```

## Neutral Actions

```text
Thinking
Standing
Sitting_down
Standing_up
Open_Door
Close_Door
```

The Rule Engine also considers object and pose information as additional evidence.

Output:

```text
outputs/decision/rule_engine.csv
```

---

# 20. Temporal Smoothing

Frame-level activity decisions can change temporarily because of recognition or detection noise.

Temporal smoothing is used to reduce unstable short-term changes.

Final timeline output:

```text
outputs/decision/final_timeline.csv
```

Final activity states:

```text
WORKING
NOT_WORKING
NEUTRAL
```

---

# 21. Productivity Report

The productivity stage calculates statistics from the final timeline.

The report includes information such as:

- Person ID
- Total frames
- Working frames
- Not Working frames
- Neutral frames
- Working Ratio
- Productivity Percentage

Output:

```text
outputs/productivity/productivity_report.csv
```

Current overall statistics:

```text
Tracked person IDs: 5

Total frames:       2,356,440
Working frames:     2,009,600
Not Working frames:   134,960
Neutral frames:       211,880

Working Ratio:       85.28%
Productivity:        93.71%
```

## Important Metric Difference

The following values are different metrics:

```text
SlowFast Accuracy: 77.88%
Working Ratio:     85.28%
Productivity:      93.71%
```

They must not be treated as equivalent.

In particular:

```text
Productivity 93.71% != Accuracy 93.71%
```

and:

```text
Working Ratio 85.28% != Accuracy 85.28%
```

---

# 22. Visualization

The project includes a visualization stage that renders the analysis results on the original video.

The visualization can contain information such as:

- Person tracking
- Stable Person ID
- Activity state
- Timeline information
- Productivity information

Example output:

```text
outputs/visualization/s_018_visualization.mp4
```

---

# 23. Current Output Statistics

The completed pipeline produced the following outputs.

## SlowFast

```text
235,644 prediction rows
```

## Object Context

```text
1,095,636 records
```

## Pose Context

```text
239,068 records
```

## Fusion

```text
235,644 rows
```

## Rule Engine

```text
WORKING:      197,702
NOT_WORKING:   14,774
NEUTRAL:       23,168
```

## Final Timeline

```text
WORKING:      200,960
NOT_WORKING:   13,496
NEUTRAL:       21,188
```

---

# 24. Main Output Files

```text
outputs/
|
+-- tracking/
|   +-- *_tracking.csv
|
+-- action/
|   +-- slowfast_predictions_all.csv
|   +-- all_label_person_candidates.csv
|   +-- label_person_candidates.csv
|   +-- label_person_mapping.csv
|
+-- objects/
|   +-- object_context.csv
|
+-- pose/
|   +-- pose_context.csv
|
+-- fusion/
|   +-- fusion.csv
|
+-- decision/
|   +-- rule_engine.csv
|   +-- final_timeline.csv
|
+-- productivity/
|   +-- productivity_report.csv
|
+-- visualization/
|   +-- s_018_visualization.mp4
|
+-- evaluation/
|   +-- evaluation_sample.csv
```

---

# 25. Evaluation

The SlowFast evaluation is performed with:

```powershell
cd "D:\SlowFast_Project"

python "D:\SlowFast_Project\train model slowfast\evaluate.py"
```

The evaluation uses:

```text
Model:
D:\SlowFast_Project\models\best_model.pth
```

and:

```text
Labels:
D:\SlowFast_Project\data\labels
```

The current evaluation result is:

```text
Accuracy : 0.7788
Precision: 0.7522
Recall   : 0.7630
F1 Score : 0.7408
```

Equivalent percentages:

```text
Accuracy : 77.88%
Precision: 75.22%
Recall   : 76.30%
F1 Score : 74.08%
```

---

# 26. Ground Truth Limitation

The available label files contain:

```text
Class_ID
Start_Frame
End_Frame
```

They do not contain:

```text
Person_ID
```

Therefore, they represent action annotations rather than independent person-level final activity labels.

The labels are appropriate for evaluating the SlowFast action-recognition task.

However, they are not sufficient by themselves to calculate a validated end-to-end Accuracy for:

```text
WORKING
NOT_WORKING
NEUTRAL
```

For a final end-to-end Accuracy, an independent Ground Truth dataset with person-level final activity labels would be required.

---

# 27. Important Interpretation of Results

The most important validated model metric currently available is:

```text
SlowFast Action Accuracy = 77.88%
```

This means the action-recognition model achieved 77.88% accuracy on the evaluated action samples.

The following values describe different aspects of the complete system:

```text
SlowFast Action Accuracy = 77.88%
Working Ratio             = 85.28%
Productivity              = 93.71%
```

Only the first value is an evaluated classification Accuracy.

The productivity and working-ratio values are system-generated statistics based on the final decisions.

---

# 28. Project Status

| Stage | Status |
|---|---|
| Person Detection | Complete |
| Person Tracking | Complete |
| Stable Person ID | Complete |
| 10-Frame Person Sequence | Complete |
| SlowFast Action Recognition | Complete |
| SlowFast Evaluation | Complete |
| Object Detection | Complete |
| Object-Person Association | Complete |
| Pose Estimation | Complete |
| Feature Fusion | Complete |
| Rule Engine | Complete |
| Temporal Smoothing | Complete |
| Productivity Report | Complete |
| Visualization | Complete |
| Independent End-to-End Ground Truth Evaluation | Not finalized |

---

# 29. Limitations

## Person Tracking

Tracking can be affected by:

- Occlusion
- Detection failures
- Similar-looking people
- Fast movement
- Long periods without detection

## Action Recognition

The current SlowFast Action Accuracy is:

```text
77.88%
```

The confusion matrix shows that some action classes are confused with other classes.

## Rule Engine

The final activity state depends on predefined rules and weights.

Changing these rules can change the final activity distribution and productivity statistics.

## Productivity

The reported productivity percentage is calculated from the system's own final decisions.

It is not an independently validated measurement of real-world employee productivity.

## End-to-End Accuracy

An independent person-level Ground Truth for the final Working / Not Working / Neutral decision has not been finalized.

Therefore, no end-to-end final Accuracy is claimed.

---

# 30. Requirements

Current environment versions:

```text
numpy==2.2.6
pandas==2.3.3
opencv-python==4.11.0.86
ultralytics==8.3.189
torch==2.13.0+cpu
torchvision==0.28.0+cpu
torchaudio==2.11.0+cpu
pytorchvideo==0.1.5
mediapipe==0.10.32
scipy==1.15.3
PyYAML==6.0.3
tqdm==4.67.3
```

---

# 31. Final Summary

This project implements a multi-stage Computer Vision system for employee activity recognition and productivity analysis.

The final pipeline is:

```text
Video
  ↓
YOLO11 Person Detection
  ↓
BoT-SORT Tracking
  ↓
Stable Person ID
  ↓
10-Frame Person Sequence
  ↓
SlowFast Action Recognition
  ↓
Object Detection
  ↓
MediaPipe Pose
  ↓
Feature Fusion
  ↓
Rule Engine
  ↓
Temporal Smoothing
  ↓
WORKING / NOT_WORKING / NEUTRAL
  ↓
Productivity Report
  ↓
Visualization
```

## Final Validated SlowFast Metrics

```text
Accuracy : 77.88%
Precision: 75.22%
Recall   : 76.30%
F1 Score : 74.08%
```

The project currently has a completed processing pipeline and a validated SlowFast action-recognition evaluation.

The final Working / Not Working / Neutral stage does not currently have an independent person-level Ground Truth dataset, so an end-to-end classification Accuracy is not claimed.

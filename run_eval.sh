#!/bin/bash

# Inference on source test
python inference.py --weights yolov8l.pt --checkpoint results/results/best-full-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_best_full_source --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/full-yolov8/weights/best-full-v8.pt --source source_test/source_test/val/images --output predicts_full_source --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/nogrl-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_nogrl_source --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/freeze_teacher-yolov8/weights/best_freeze_full_v8.pt --source source_test/source_test/val/images --output predicts_freeze_full_source --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/freeze_teacher-nogrl-yolov8/weights/best_freeze_nogrl-v8.pt --source source_test/source_test/val/images --output predicts_freeze_nogrl_source --conf-thres 0.25

python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_source_only_100/weights/best.pt --source source_test/source_test/val/images --output predicts_baseline_source_only_100_source --conf-thres 0.25 
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_source_only_200/weights/best.pt --source source_test/source_test/val/images --output predicts_baseline_source_only_200_source --conf-thres 0.25 
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_target_only_100/weights/best.pt --source source_test/source_test/val/images --output predicts_baseline_target_only_100_source --conf-thres 0.25 
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_target_only_200/weights/best.pt --source source_test/source_test/val/images --output predicts_baseline_target_only_200_source --conf-thres 0.25 

# # Inference on target test
python inference.py --weights yolov8l.pt --checkpoint results/results/best-full-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_best_full_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/full-yolov8/weights/best-full-v8.pt --source target_test/target_test/val/images --output predicts_full_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/nogrl-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_nogrl_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/freeze_teacher-yolov8/weights/best_freeze_full_v8.pt --source target_test/target_test/val/images --output predicts_freeze_full_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/freeze_teacher-nogrl-yolov8/weights/best_freeze_nogrl-v8.pt --source target_test/target_test/val/images --output predicts_freeze_nogrl_target --conf-thres 0.25

python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_source_only_100/weights/best.pt --source target_test/target_test/val/images --output predicts_baseline_source_only_100_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_source_only_200/weights/best.pt --source target_test/target_test/val/images --output predicts_baseline_source_only_200_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_target_only_100/weights/best.pt --source target_test/target_test/val/images --output predicts_baseline_target_only_100_target --conf-thres 0.25
python inference.py --weights yolov8l.pt --checkpoint results/results/baseline/baseline_target_only_200/weights/best.pt --source target_test/target_test/val/images --output predicts_baseline_target_only_200_target --conf-thres 0.25

# Evaluation on source test
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_best_full_source
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_full_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_nogrl_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_freeze_full_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_freeze_nogrl_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_source_only_100_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_source_only_200_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_target_only_100_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_target_only_200_source 

# Evaluation on target test
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_best_full_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_full_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_nogrl_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_freeze_full_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_freeze_nogrl_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_source_only_100_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_source_only_200_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_target_only_100_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_target_only_200_target 

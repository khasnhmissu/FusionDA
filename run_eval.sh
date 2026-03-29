#!/bin/bash

# Inference on source test
python inference.py --weights yolov8l.pt --checkpoint results_explainable/full-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_full_source --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/nogrl-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_nogrl_source --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/freeze_teacher-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_freeze_full_source --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/freeze_teacher-nogrl-yolov8/weights/best.pt --source source_test/source_test/val/images --output predicts_freeze_nogrl_source --conf-thres 0.25 --imgsz 1024
# python inference_base.py --weights yolov8l.pt --checkpoint result_explainable/baseline-v8/baseline_source_only/weights/source.pt --source source_test/source_test/val/images --output predicts_baseline_source_only_source --conf-thres 0.25 --imgsz 384
# python inference_base.py --weights yolov8l.pt --checkpoint result_explainable/baseline-v8/baseline_target_only/weights/target.pt --source source_test/source_test/val/images --output predicts_baseline_target_only_source --conf-thres 0.25 --imgsz 384

# # Inference on target test
python inference.py --weights yolov8l.pt --checkpoint results_explainable/full-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_full_target --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/nogrl-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_nogrl_target --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/freeze_teacher-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_freeze_full_target --conf-thres 0.25 --imgsz 1024
python inference.py --weights yolov8l.pt --checkpoint results_explainable/freeze_teacher-nogrl-yolov8/weights/best.pt --source target_test/target_test/val/images --output predicts_freeze_nogrl_target --conf-thres 0.25 --imgsz 1024
# python inference_base.py --weights yolov8l.pt --checkpoint results/results/baseline-v8/baseline_source_only/weights/source.pt --source target_test/target_test/val/images --output predicts_baseline_source_only_target --conf-thres 0.25 --imgsz 384
# python inference_base.py --weights yolov8l.pt --checkpoint results/results/baseline-v8/baseline_target_only/weights/target.pt --source target_test/target_test/val/images --output predicts_baseline_target_only_target --conf-thres 0.25 --imgsz 384

# Evaluation on source test
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_full_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_nogrl_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_freeze_full_source 
python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_freeze_nogrl_source 
# python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_source_only_source 
# python map_evaluation.py --labels source_test/source_test/val/labels --predicts predicts_baseline_target_only_source 

# Evaluation on target test
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_full_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_nogrl_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_freeze_full_target 
python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_freeze_nogrl_target 
# python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_source_only_target 
# python map_evaluation.py --labels target_test/target_test/val/labels --predicts predicts_baseline_target_only_target 

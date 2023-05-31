cd ~/frameworks/yolov5
python export.py --data /WORKING/training/office_tracking/cfg_train/ds_yolov5.yaml  --weights /DS/Datasets/CH_custom/VOC/Human/office_people_counter/trained_models/20230209/yolov5s/best.pt --include onnx

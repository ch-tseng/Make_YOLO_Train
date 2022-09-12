'''
pip install nvidia-pyindex
pip install onnx_graphsurgeon
pip install onnx>=1.9.0
pip install onnxruntime
pip install --ignore-installed PyYAML
pip install --upgrade nvidia-tensorrt
pip install pycuda
pip install onnxsim

cd ~/frameworks/
git clone https://github.com/Linaom1214/TensorRT-For-YOLO-Series.git 
cd TensorRT-For-YOLO-Series
'''

cd ~/frameworks/yolov7
python export.py --weights /DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/v1_20220909/yolov7/yolov7.pt \
                 --device 0 --grid --simplify --include-nms


cd ~/frameworks/TensorRT-For-YOLO-Series
python export.py -o /DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/v1_20220909/yolov7/yolov7.onnx \
                 -e ./crowded_human.trt -p fp16


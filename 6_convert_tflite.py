'''
cd ~/frameworks
git clone https://github.com/MPolaris/onnx2tflite.git
cd onnx2tflite

#int8
python converter.py \
    --weights /DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/ONNX/yolov5s.onnx \
    --outpath "/DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/TFLite/" \
    --formats "tflite" \
    --int8 \
    --imgroot /DS/Datasets/CH_custom/VOC/Human/CrowdedHuman/trained_weights/TFLite/test_images  \
    --int8mean 0 0 0 --int8std 1 1 1

'''

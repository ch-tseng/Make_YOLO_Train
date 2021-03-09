import os,sys

classList = { "forklift":0 }
cfgFolder = "/WORK1/MyProjects/for_Sale/forklift/train_models/yolo_models/"
dark_home = "~/frameworks/darknet.v4"
yolov5_home = "~/frameworks/yolov5"

yolo_config = {
    'numBatch': 8,
    'numSubdivision': 4,
    '416': "28, 57,  49,102,  87,143,  57,223,  92,300, 137,219, 135,356, 196,358, 292,394",
    '512': "35, 72,  59,137, 117, 99, 102,190,  82,318, 163,275, 134,421, 210,435, 340,477",
    '608': "40, 84,  72,142,  75,294, 128,209, 121,411, 201,322, 172,502, 258,521, 409,570",
    '640': "42, 89,  75,152,  83,326, 134,221, 138,449, 214,341, 204,546, 302,553, 449,606",
}

yolotiny_config = {
    'numBatch': 32,
    'numSubdivision': 2,
    '320': "27, 53,  49,104,  59,213,  97,157, 109,264, 193,294",
    '416': "35, 70,  64,137,  80,283, 126,203, 146,341, 255,386",
}

#---------------------------------------------------------------------

classNum = len(classList)
filterNum = (classNum + 5) * 3

#default size:
#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512
cfgs = {
    "yolov3": ["cfg/yolov3.cfg", "pretrained/darknet53.conv.74", 608],
    "yolov3-tiny": ["cfg/yolov3-tiny.cfg", "pretrained/yolov3-tiny.conv.15", 416],
    "yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137", 608],
    "yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29", 416],
    "yolo-fastest": ["cfg/yolo-fastest.cfg", "pretrained/yolo-fastest.weights", 320],
    "yolo-fastest-xl": ["cfg/yolo-fastest-xl.cfg", "pretrained/yolo-fastest-xl.weights", 320],
    "yolov4x-mish": ["cfg/yolov4x-mish.cfg", "pretrained/yolov4x-mish.conv.166", 640],
    "yolov4-csp": ["cfg/yolov4-csp.cfg", "pretrained/yolov4-csp.conv.142", 512],
    "yolov5s": ["cfg/yolov5s.yaml", "pretrained/yolov5s.pt", 640],
    "yolov5m": ["cfg/yolov5m.yaml", "pretrained/yolov5m.pt", 640],
    "yolov5l": ["cfg/yolov5l.yaml", "pretrained/yolov5l.pt", 640],
    "yolov5x": ["cfg/yolov5x.yaml", "pretrained/yolov5x.pt", 640],
}

pwd = os.getcwd()

#make dataset yaml for YOLOV5
with open('cfg/data_yolov5.yaml') as file:
    dataset_content = file.read()
file.close

class_txt = '['
for i, cname in enumerate(classList):
    class_txt += "'{}'".format(cname)
    if i<(len(classList)-1): class_txt += ', '
class_txt += ']'

dataset_content = dataset_content.replace("{TRAIN_LIST}", os.path.join(cfgFolder,'train.txt'))
dataset_content = dataset_content.replace("{TEST_LIST}", os.path.join(cfgFolder,'test.txt'))
dataset_content = dataset_content.replace("{CLASSES}", str(classNum))
dataset_content = dataset_content.replace("{CLASS_LIST}", class_txt)

file = open(os.path.join(cfgFolder, 'ds_yolov5.yaml'), "w")
file.write(dataset_content)
file.close
#---- end

for cfg_name in cfgs:
    with open(cfgs[cfg_name][0]) as file:
        file_content = file.read()
    file.close

    if(cfg_name[:6] == 'yolov5'):
        print("TEST", str(cfgs[cfg_name][2]))
        anchors = yolo_config[str(cfgs[cfg_name][2])]
        anch_list = anchors.split(',')

        anchors1, anchors2, anchors3 = "", "", ""
        for a in range(0,3):
            anchors1 += anch_list[a]
            if a<2: anchors1 += ','
        for a in range(3,6):
            anchors2 += anch_list[a]
            if a<5: anchors2 += ','
        for a in range(6,9):
            anchors3 += anch_list[a]
            if a<8: anchors3 += ','

        file_updated = file_content.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py --data {}  --cfg {} --batch {} --epochs 300 --weights {}".format( \
            yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), os.path.join(cfgFolder,cfg_name+'.yaml'), \
            yolo_config["numBatch"],os.path.join(pwd,cfgs[cfg_name][1]))


    else:
        if(cfg_name in ["yolov3-tiny", "yolov4-tiny", "yolo-fastest", "yolo-fastest-xl"]):
            batch = yolotiny_config['numBatch']
            div = yolotiny_config['numSubdivision']
            anch = yolotiny_config[str(cfgs[cfg_name][2])]

        else:
            batch = yolo_config['numBatch']
            div = yolo_config['numSubdivision']
            anch = yolo_config[str(cfgs[cfg_name][2])]

        file_updated = file_content.replace("{BATCH}", str(batch))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(div))
        file_updated = file_updated.replace("{SIZE}", str(cfgs[cfg_name][2]))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHORS}", anch)

        cfg_file = cfg_name +'.cfg'

        exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show -mjpeg_port 8090 -gpus 0".format(\
            dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]))

    file = open(os.path.join(cfgFolder, cfg_file), "w")
    file.write(file_updated)
    file.close

    print("-----------------------------------------")
    print(" Command for training {} model".format(cfg_name.upper()))
    print("-----------------------------------------")
    print(exec_cmd)
    print('')

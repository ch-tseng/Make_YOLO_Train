import os,sys

cfgFolder = "/WORK1/dataset/crowd_human_water/try2/cfg_train"

classList = { "person_head":0, "person_vbox":1 }

'''
classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''
dark_home = "/home/chtseng/frameworks/darknet.v4"
yolov5_home = "/home/chtseng/frameworks/yolov5"

yolo_config = {
    'numBatch': 120,
    'numSubdivision': 40,
    '416': "12, 13,  27, 40,  71, 43,  57,103, 151, 88, 107,185, 276,166, 186,300, 339,327",
    '512': "4, 17,  42, 47,  62,109, 132, 71, 110,205, 215,145, 211,350, 372,222, 404,418",
    '608': "17, 21,  50, 56,  76,133, 156, 84, 133,249, 255,171, 255,418, 442,263, 482,496",
    '640': "18, 22,  52, 58,  78,136, 165, 89, 137,256, 268,180, 264,438, 465,277, 505,522",
    '1536': "31, 36,  53,101, 104, 58, 119,160,  72,267, 222,101, 179,293, 296,195, 218,565, 343,374, 584,228, 405,637, 624,430, 421,1059, 1108,477, 737,724, 670,1207, 1261,829, 962,1229, 1374,1345"
}

yolotiny_config = {
    'numBatch': 32,
    'numSubdivision': 2,
    '320': "10, 12,  34, 38,  70, 83, 181,101, 111,191, 240,238",
    '416': "14, 16,  43, 50,  95,105, 138,245, 247,142, 308,316",
}

#---------------------------------------------------------------------

classNum = len(classList)
filterNum = (classNum + 5) * 3

#default size:
#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
cfgs = {
    "yolov3": ["cfg/yolov3.cfg", "pretrained/darknet53.conv.74", 608],
    "yolov3-tiny": ["cfg/yolov3-tiny.cfg", "pretrained/yolov3-tiny.conv.15", 416],
    "yolov3-spp": ["cfg/yolov3-spp.cfg", "pretrained/yolov3-spp.weights", 608],
    "yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137", 608],
    "yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29", 416],
    "yolo-fastest": ["cfg/yolo-fastest.cfg", "pretrained/yolo-fastest.weights", 320],
    "yolo-fastest-xl": ["cfg/yolo-fastest-xl.cfg", "pretrained/yolo-fastest-xl.weights", 320],
    "yolov4x-mish": ["cfg/yolov4x-mish.cfg", "pretrained/yolov4x-mish.conv.166", 640],
    "yolov4-csp": ["cfg/yolov4-csp.cfg", "pretrained/yolov4-csp.conv.142", 512],
    "yolov4-cspx-p7": ["cfg/cspx-p7-mish.cfg", "pretrained/cspx-p7-mish_hp.344.conv", 1536],
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
        for a in range(0,6):
            anchors1 += anch_list[a]
            if a<6: anchors1 += ','
        for a in range(6,12):
            anchors2 += anch_list[a]
            if a<12: anchors2 += ','
        for a in range(12,18):
            anchors3 += anch_list[a]
            if a<18: anchors3 += ','

        file_updated = file_content.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py \\\n    --data {} \\\n    --cfg {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --weights {}".format( \
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

        exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
            dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]))

    file = open(os.path.join(cfgFolder, cfg_file), "w")
    file.write(file_updated)
    file.close

    print("-----------------------------------------")
    print(" Command for training {} model".format(cfg_name.upper()))
    print("-----------------------------------------")
    print(exec_cmd)
    print('')

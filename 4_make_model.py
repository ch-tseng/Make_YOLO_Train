import os,sys

classList = { "person_head":0, "person_vbox":1 }

cfgFolder = "/WORKING/modelSale/crowd_human/cfg_train"
'''
classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''
dark_home = "/home/chtseng/frameworks/darknet"
yolofastest_home = "/home/chtseng/frameworks/Yolo-Fastest"
yolov5_home = "/home/chtseng/frameworks/yolov5"

yolo_config = {
    'numBatch': 120,
    'numSubdivision': 40,
    '416': "7,  5,   8, 20,  24,  9,  17, 48,  59, 19,  32,106, 125, 38,  73,194, 231,101",
    '512': "9,  7,  28, 11,  12, 30,  62, 21,  25, 77, 129, 37,  56,174, 223, 76, 246,251",
    '608': "10,  8,  33, 13,  14, 35,  75, 25,  30, 92, 154, 44,  67,207, 265, 90, 292,299",
    '640': "12,  8,  12, 30,  38, 15,  25, 72,  92, 29,  49,162, 195, 59, 110,296, 357,158",
    '768': "13, 10,  42, 16,  17, 45,  95, 32,  38,116, 195, 56,  84,262, 336,114, 369,378",
    '960': "16, 13,  52, 20,  22, 55, 117, 39,  47,145, 242, 70, 105,327, 418,142, 460,471",
    '1536': "19, 15,  47, 20,  21, 48,  68, 37, 125, 28,  37,102, 114, 58,  73,142, 228, 50,  58,260, 190, 97, 376, 84, 110,376, 304,184, 557,135, 186,597, 693,241, 364,825, 968,405, 1000,972"
}

yolotiny_config = {
    'numBatch': 32,
    'numSubdivision': 2,
    '320': "7,  5,  26,  9,   9, 26,  74, 22,  25, 82, 142, 89",
    '416': "9,  7,  34, 12,  12, 34,  97, 29,  33,107, 185,115",
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
    "yolo-fastest": ["cfg/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest.conv.109", 320],
    "yolo-fastest-xl": ["cfg/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest-xl.conv.109", 320],
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

tfile = open( os.path.join(cfgFolder, 'train_cmd.txt'), 'w')

for cfg_name in cfgs:
    print('test', cfg_name)
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

        exec_cmd = " cd {}\n python train.py \\\n    --data {} \\\n    --cfg {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --noautoanchor    \\\n    --weights {}".format( \
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

        if (cfg_name in ["yolo-fastest", "yolo-fastest-xl"]):
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
                yolofastest_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]))

        else:
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

    tfile.write("---------------------------------------------------------------------\n")
    tfile.write(" [{} model] \n".format(cfg_name.upper()))
    tfile.write(exec_cmd + '\n\n')

tfile.close()

import os,sys

classList = { "forklift":0 }
cfgFolder = "/WORKING/modelSale/forklift/cfg_train"

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
    '416': "7,  7,  12, 27,  27, 12,  53, 26,  26, 53,  97, 51,  52, 98, 147,145, 275,284",
    '512': "9,  9,  33, 15,  15, 34,  65, 32,  32, 66, 120, 63,  64,121, 181,178, 338,350",
    '608': "11, 10,  18, 40,  40, 18,  77, 38,  38, 78, 142, 75,  75,143, 215,211, 402,415",
    '640': "11, 11,  42, 19,  19, 42,  81, 40,  40, 82, 150, 79,  80,151, 226,222, 422,437",
    '1536': "17, 15,  23, 52,  57, 25,  43, 91, 102, 47,  79,125,  58,191, 146, 84, 216, 66, 109,205, 193,135, 334,131, 145,331, 345,223, 239,382, 635,273, 359,695, 717,481, 709,1299, 1334,771"
}

yolotiny_config = {
    'numBatch': 32,
    'numSubdivision': 2,
    '320': "8,  8,  16, 33,  35, 17,  38, 73,  75, 39, 136,137",
    '416': "10, 11,  42, 21,  22, 46,  93, 48,  52, 98, 179,175",
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

tfile = open( os.path.join(cfgFolder, 'train_cmd.txt'), 'w')

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

    tfile.write("---------------------------------------------------------------------\n")
    tfile.write(" [{} model] \n".format(cfg_name.upper()))
    tfile.write(exec_cmd + '\n\n')

tfile.close()

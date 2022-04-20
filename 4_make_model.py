import os,sys

classList = { 'person_head':0, 'person_vbox':1 }
cfgFolder = "/WORKING/modelSale/Crowd_human/dataset/aug_20220413/cfg_train"
project_name = 'Crowd_human'

'''
classList = { 'D00':0, 'D10':1, 'D20':2, 'D21':3, 'D30':4 ,'D31':5, 'D40':6, 'D41':7, 'D42':8, 'D99':9 }
cfgFolder = "/WORKING/WORKS/road_defects_2022_04/aug_20220408/cfg_train"

classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''
dark_home = "/home/chtseng/frameworks/darknet"
yolofastest_home = "/home/chtseng/frameworks/darknet"
yolov5_home = "/home/chtseng/frameworks/yolov5"
yolor_home = "/home/chtseng/frameworks/yolor"

cfgs = {
     "yolov3": ["cfg/yolov3/yolov3.cfg", "pretrained/yolov3/darknet53.conv.74", '608_9', 64, 32, 3],
     "yolov3-tiny": ["cfg/yolov3/yolov3-tiny.cfg", "pretrained/yolov3/yolov3-tiny.conv.15", '416_6', 70, 2, 3],
     "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
     "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 72, 1, 3],
     "yolov4-p6": ["cfg/yolov4/yolov4-p6.cfg", 'pretrained/yolov4-p6.conv.289', '1280_12', 64, 64, 4],
     "yolo-fastest": ["cfg/yolo-fastest/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest/yolo-fastest.conv.109", '320_6', 160, 2, 3],
     "yolo-fastest-xl": ["cfg/yolo-fastest/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest/yolo-fastest-xl.conv.109", '320_6', 120, 2, 3],
     "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', 64, 1, 3],
     "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1, 3],
     "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', 64, 1, 3],
     "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', 64, 1, 3],
     "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', 64, 1, 3],
     "yolov5s-p6_1280": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '1280_12', 64, 1, 4],
     "yolov5x-p6_1280": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 48, 1, 4],
     "yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 64, 16, 3],
     #"yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 66, 33, 3],
     "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor-p6.pt", "1280_12", 66, 66, 3],
     "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor-w6.pt", "1280_12", 66, 66, 3],
     "yolor_yolov4_p7": ["cfg/yolor/yolov4_p7.cfg", '', "1536_20", 66,66, 4]
     #"yolor_yolov4_p6": ["cfg/yolor/yolor_p6.cfg", '', "640_12", 66, 66, 4],
     #"yolor_yolov4_p7": ["cfg/yolor/yolor_p6.cfg", '', "640_20", 66, 66, 4]
}

yolo_config = {
    '320_6': "5,  8,  12, 30,  36, 14,  27, 89, 132, 40,  80,196",
    '416_6': "7, 10,  17, 34,  91, 29,  32,100,  69,230, 236,151",
    '320_9': "4,  6,   8, 20,  24, 10,  17, 45,  66, 23,  29, 98, 167, 52,  53,182, 143,225",
    '416_9': "5,  7,  11, 26,  31, 13,  21, 59,  86, 30,  38,128, 217, 68,  69,237, 186,292",
    '512_9': "7,  9,  13, 31,  38, 16,  26, 72, 106, 37,  47,157, 267, 83,  85,292, 230,359",
    '608_9': "8, 11,  16, 37,  45, 19,  31, 86, 126, 44,  55,187, 318, 99, 101,347, 272,427",
    '640_9': "8, 11,  17, 39,  47, 20,  33, 90, 133, 46,  58,197, 334,104, 106,365, 287,449",
    '960_9': "13, 17,  25, 59,  71, 30,  50,135, 199, 69,  87,295, 501,156, 159,547, 430,674",
    '1280_9': "17, 23,  33, 79,  95, 40,  66,181, 266, 92, 116,394, 668,208, 212,730, 574,899",
    '640_12': "7, 10,  14, 32,  37, 16,  26, 67,  90, 35,  41,133, 217, 61,  89,161,  61,284, 123,398, 396,132, 305,467",
    '960_12': "11, 15,  21, 48,  55, 25,  39,100, 135, 52,  62,199, 326, 92, 133,242,  92,425, 185,597, 593,198, 458,701",
    '1280_12': "15, 20,  28, 64,  74, 33,  53,133, 180, 69,  82,266, 435,122, 178,323, 123,567, 246,796, 791,265, 610,934",
    '1536_12': "18, 24,  34, 77,  89, 39,  63,160, 216, 83,  99,319, 521,147, 214,387, 147,680, 296,955, 949,317, 732,1122",
    '1536_20': "14, 18,  23, 47,  63, 30,  37, 90,  65,119, 139, 56,  49,216,  98,187, 278, 95,  84,371, 154,287, 524,151, 129,615, 217,466, 212,914, 887,232, 384,615, 380,1158, 1108,462, 844,1179"
}

#yolotiny_config = {
#    '320': "7, 17,  11, 23,  20, 37,  53, 39,  85, 48, 170,105",
#    '416': "9, 22,  15, 30,  27, 48,  68, 51, 110, 62, 221,137",
#}

#---------------------------------------------------------------------

classNum = len(classList)
#filterNum = (classNum + 5) * 3

#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
#    [CFG FILE, PRE-TRAINED WEIGHTS, SIZE, BATCH, DIVISION-BATCH, MASKS]
cfgs_total = {
    "yolov3": ["cfg/yolov3/yolov3.cfg", "pretrained/yolov3/darknet53.conv.74", '608_9', 64, 32, 3],
    "yolov3-tiny": ["cfg/yolov3/yolov3-tiny.cfg", "pretrained/yolov3/yolov3-tiny.conv.15", '416_6', 66, 2, 3],
    "yolov3-spp": ["cfg/yolov3/yolov3-spp.cfg", "pretrained/yolov3/yolov3-spp.weights", '608_9', 64, 32, 3],
    "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
    "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 72, 1, 3],
    "yolo-fastest": ["cfg/yolo-fastest/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest/yolo-fastest.conv.109", '320_6', 160, 2, 3],
    "yolo-fastest-xl": ["cfg/yolo-fastest/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest/yolo-fastest-xl.conv.109", '320_6', 120, 2, 3],
    "yolov4x-mish": ["cfg/yolov4/yolov4x-mish.cfg", '', '640_9', 64, 64, 3],
    "yolov4-csp": ["cfg/yolov4/yolov4-csp.cfg", '', '640_9', 64, 64, 3],
    "yolov4-p5": ["cfg/yolov4/yolov4-p5.cfg", 'pretrained/yolov4-p5.conv.232', '896_12', 64, 64, 4],
    "yolov4-p6": ["cfg/yolov4/yolov4-p6.cfg", 'pretrained/yolov4-p6.conv.289', '1280_12', 64, 64, 4],
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', -1, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', -1, 1, 3],
    "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', -1, 1, 3],
    "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', -1, 1, 3],
    "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', -1, 1, 3],
    "yolov5s-p6_640": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '640_12', -1, 1, 3],
    "yolov5s-p6_960": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '960_12', -1, 1, 3],
    "yolov5s-p6": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '1280_12', -1, 1, 3],
    "yolov5m-p6": ["cfg/yolov5/yolov5m6.yaml", "yolov5m6.pt", '1280_12', -1, 1, 3],
    "yolov5l-p6": ["cfg/yolov5/yolov5l6.yaml", "yolov5l6.pt", '1280_12', -1, 1, 3],
    "yolov5x-p6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 32, 1, 3],
    "yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 32, 16, 3],
    "yolor_csp_star": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp_star.pt", "640_9", 32, 16, 3],
    "yolor_csp_x": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x.pt", "640_9", 64, 22, 3],
    "yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 64, 33, 3],
    "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor_p6.pt", "1280_12", 64, 66, 3],
    "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor_w6.pt", "1280_12", 64, 66, 3],
    "yolor_yolov4_csp":  ["cfg/yolor/yolov4_csp.cfg", '', "640_9", 66, 33, 3],
    "yolor_yolov4_csp_x":  ["cfg/yolor/yolov4_csp_x.cfg", '', "640_9", 66, 66, 3],
    "yolor_yolov4_p6": ["cfg/yolor/yolov4_p6.cfg", '', "1280_16", 66,66, 4],
    "yolor_yolov4_p7": ["cfg/yolor/yolov4_p7.cfg", '', "1536_20", 66,66, 4]
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

#make dataset yaml for YOLOR
with open('cfg/yolor_data.yaml') as file:
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

file = open(os.path.join(cfgFolder, 'ds_yolor.yaml'), "w")
file.write(dataset_content)
file.close
#---- end


tfile = open( os.path.join(cfgFolder, 'train_cmd.txt'), 'w')

for cfg_name in cfgs:
    if(cfg_name[:6] == 'yolov5'):
        anchors = yolo_config[cfgs[cfg_name][2]]
        anch_list = anchors.split(',')

        anchors1, anchors2, anchors3, anchors4 = "", "", "", ""
        for a in range(0,6):
            anchors1 += anch_list[a]
            if a<6: anchors1 += ','
        for a in range(6,12):
            anchors2 += anch_list[a]
            if a<12: anchors2 += ','
        for a in range(12,18):
            anchors3 += anch_list[a]
            if a<18: anchors3 += ','
        if len(anch_list) > 18:
            for a in range(18,24):
                anchors4 += anch_list[a]
                if a<24: anchors4 += ','

        with open(cfgs[cfg_name][0]) as file:
            file_content = file.read()
        file.close

        #file_content = open( cfgs[cfg_name][0], 'w')

        file_updated = file_content.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        if len(anch_list) >= 18:
            file_updated = file_updated.replace("{ANCHOR4}", str(anchors4))

        cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py \\\n  --project {}_{} --name {}  --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --weights {}".format( \
            yolov5_home, project_name, cfg_name, cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2].split('_')[0], \
            cfgs[cfg_name][3],cfgs[cfg_name][1])


    else:
        with open(cfgs[cfg_name][0]) as file:
            file_content = file.read()
        file.close

        batch = cfgs[cfg_name][3]
        div = cfgs[cfg_name][4]

        #if(cfg_name in ["yolov3-tiny", "yolov4-tiny", "yolo-fastest", "yolo-fastest-xl"]):
        #    anch = yolotiny_config[str(cfgs[cfg_name][2])]

        #else:
        anch = yolo_config[str(cfgs[cfg_name][2])]
        filterNum = (classNum + 5) * cfgs[cfg_name][5]

        file_updated = file_content.replace("{BATCH}", str(batch))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(div))
        file_updated = file_updated.replace("{SIZE}", str(cfgs[cfg_name][2].split('_')[0]))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHORS}", anch)

        cfg_file = cfg_name +'.cfg'

        if (cfg_name in ["yolo-fastest", "yolo-fastest-xl"]):
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
                yolofastest_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,'yolo-fastest',cfg_name+'.cfg'), os.path.join(pwd,'yolo-fastest',cfgs[cfg_name][1]))

        elif(cfg_name[:5] == 'yolor'):
            if cfgs[cfg_name][2].split('_')[0] in ['640','512']:
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.640.yaml')
            elif cfgs[cfg_name][2].split('_')[0] == '1280':
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.1280.yaml')

            if cfgs[cfg_name][1] != '':
                weights_file = os.path.join(pwd,cfgs[cfg_name][1])
            else:
                weights_file = ''

            exec_cmd = "cd {}\\\n python train.py --batch-size {} --img {} {} --data {} --cfg {} --weights '{}' --device 0 --name {} --hyp {} --epochs {}".format(\
                yolor_home, cfgs[cfg_name][3], cfgs[cfg_name][2].split('_')[0], cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, 'ds_yolor.yaml'), \
                os.path.join(cfgFolder,cfg_name+'.cfg'), weights_file, cfg_name, hyp_file, 300)

        else:
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
                dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]) )


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

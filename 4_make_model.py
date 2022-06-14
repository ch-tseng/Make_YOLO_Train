import os,sys

project_name = "face_eyeballs"
weights_save = "/WORKING/modelSale/face_eyeballs/weights/"
classList = { '0':0, 'eye':1, 'nose':2, 'mouth':3, 'face':4, 'head':5, 'body':6 }
cfgFolder = "/WORKING/modelSale/face_mask_eyeball/aug_20220613/cfg_train"

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
    "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 64, 1, 3],
     "yolov4-p6": ["cfg/yolov4/yolov4-p6.cfg", 'pretrained/yolov4-p6.conv.289', '1280_12', 64, 64, 4],
     "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
    "yolo-fastest": ["cfg/yolo-fastest/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest/yolo-fastest.conv.109", '320_6', 160, 2, 3],
     "yolo-fastest-xl": ["cfg/yolo-fastest/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest/yolo-fastest-xl.conv.109", '320_6', 120, 2, 3],
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', -1, 1, 3],
     "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', -1, 1, 3],
    "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', -1, 1, 3],
    "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', -1, 1, 3],
    "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', -1, 1, 3],
     #"yolov5s-p6_640": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '640_12', 64, 1, 3],
     #"yolov5s-p6_960": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '960_12', 48, 1, 3],
     "yolov5x-p6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', -1, 1, 3],
     #"yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 64, 16, 3],
     #"yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 66, 33, 3],
     "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor-p6.pt", "1280_12", 66, 66, 3],
     "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor-w6.pt", "1280_12", 66, 66, 3],
     #"yolor_yolov4_p6": ["cfg/yolor/yolor_p6.cfg", '', "640_12", 66, 66, 4],
     #"yolor_yolov4_p7": ["cfg/yolor/yolor_p6.cfg", '', "640_20", 66, 66, 4]
}

yolo_config = {
    '320_6': "13,  9,  34, 25,  79, 97, 139,146, 182,239, 290,295",
    '416_6': "17, 12,  44, 33, 103,126, 181,190, 237,311, 377,383",
    '320_9': "10,  8,  28, 13,  19, 31,  48, 33,  84, 98, 108,179, 162,138, 192,247, 293,296",
    '416_9': "13, 10,  36, 17,  24, 40,  62, 43, 110,127, 140,233, 211,179, 249,322, 381,385",
    '512_9': "16, 13,  45, 21,  30, 49,  76, 53, 135,157, 172,287, 260,220, 307,396, 469,474",
    '608_9': "19, 15,  53, 24,  36, 58,  91, 63, 160,186, 204,341, 309,261, 365,470, 557,563",
    '640_9': "20, 16,  56, 26,  38, 61,  96, 67, 171,198, 258,309, 432,367, 340,549, 583,598",
    '960_9': "29, 24,  84, 39,  56, 92, 143,100, 253,294, 323,538, 487,413, 576,742, 879,889",
    '1280_9': "39, 32, 112, 51,  75,123, 191,133, 337,391, 430,717, 650,550, 768,989, 1173,1185",
    '640_12': "18, 15,  43, 23,  40, 63,  86, 36,  98, 91, 162,190, 188,333, 288,242, 298,396, 467,384, 379,580, 596,602",
    '960_12': "26, 23,  64, 35,  60, 94, 129, 54, 144,135, 246,275, 271,472, 432,377, 407,681, 643,546, 615,870, 901,897",
    '1280_12': "32, 28,  73, 42,  68,113, 141, 60, 130,174, 231,106, 317,390, 575,499, 431,742, 846,747, 712,1121, 1176,1196",
    '1536_12': "39, 34,  87, 50,  82,135, 170, 72, 156,209, 278,129, 381,468, 691,599, 518,890, 1015,896, 855,1346, 1411,1435"
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
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', 128, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1, 3],
    "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', 24, 1, 3],
    "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', 12, 1, 3],
    "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', 8, 1, 3],
    "yolov5s-p6_640": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '640_12', 48, 1, 3],
    "yolov5s-p6_960": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '960_12', 48, 1, 3],
    "yolov5s-p6": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '1280_12', 48, 1, 3],
    "yolov5m-p6": ["cfg/yolov5/yolov5m6.yaml", "yolov5m6.pt", '1280_12', 32, 1, 3],
    "yolov5l-p6": ["cfg/yolov5/yolov5l6.yaml", "yolov5l6.pt", '1280_12', 24, 1, 3],
    "yolov5x-p6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 12, 1, 3],
    "yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 64, 16, 3],
    "yolor_csp_star": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp_star.pt", "640_9", 64, 16, 3],
    "yolor_csp_x": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x.pt", "640_9", 66, 22, 3],
    "yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 66, 33, 3],
    "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor_p6.pt", "1280_12", 66, 66, 3],
    "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor_w6.pt", "1280_12", 66, 66, 3],
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
        path_project = os.path.join( weights_save, project_name )
        path_project_name = os.path.join(path_project, cfg_name)

        exec_cmd = " cd {}\n $(which python) train.py \\\n    --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --project {} \\\n    --name {} \\\n    --device {} \\\n    --weights {}".format( \
            yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2].split('_')[0], \
            cfgs[cfg_name][3], path_project, path_project_name, '{GPU}', cfgs[cfg_name][1])

        #exec_cmd += " --freeze 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

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
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port {} \\\n    -clear \\\n    -gpus {}".format(\
                yolofastest_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]), '{DARKNET_PORT}', '{GPU}')

        elif(cfg_name[:5] == 'yolor'):
            if cfgs[cfg_name][2].split('_')[0] in ['640','512']:
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.640.yaml')
            elif cfgs[cfg_name][2].split('_')[0] == '1280':
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.1280.yaml')

            if cfgs[cfg_name][1] != '':
                weights_file = os.path.join(pwd,cfgs[cfg_name][1])
            else:
                weights_file = ''

            exec_cmd = "cd {}\\\n $(which python) train.py --batch-size {} --img {} {} --data {} --cfg {} --weights '{}' --device {} --name {} --hyp {} --epochs {}".format(\
                yolor_home, cfgs[cfg_name][3], cfgs[cfg_name][2].split('_')[0], cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, 'ds_yolor.yaml'), \
                os.path.join(cfgFolder,cfg_name+'.cfg'), weights_file, '{GPU}', cfg_name, hyp_file, 300)

        else:
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port {} \\\n    -clear \\\n    -gpus {}".format(\
                dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]), '{DARKNET_PORT}', '{GPU}' )


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

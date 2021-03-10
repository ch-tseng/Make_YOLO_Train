import os, subprocess

cfgFolder = "/WORK1/MyProjects/for_Sale/forklift/train_models/yolo_models/"
dark_home = "/home/chtseng/frameworks/darknet.v4"
sizes = [ [9,416], [9,512], [9,608], [9,640], [20,1536], [6,320], [6,416] ]


darknet_path = os.path.join(dark_home, 'darknet')

for i, size in enumerate(sizes):
    print("=============================================================================================================")
    print("YOLO anchors Caculating for {}x{}, {} boxes...................................................".format(\
        sizes[i][1], sizes[i][1], sizes[i][0]))
    print("=============================================================================================================")
    subprocess.run([darknet_path, 'detector', 'calc_anchors', os.path.join(cfgFolder, 'obj.data'), '-num_of_clusters', \
        str(sizes[i][0]), '-width', str(sizes[i][1]), '-height', str(sizes[i][1]) ])




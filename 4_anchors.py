import os, subprocess
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

project_name = cfg.get("global", "project_name")
baseFolder = cfg.get("global", "baseFolder")
cfgFolder = os.path.join(baseFolder, project_name, "cfg_train")
darknet_home = cfg.get("yoloPath", "yolov4_home")
sizes = ast.literal_eval(cfg.get("models", "anchors"))
#--------------------------------------------------------
baseFolder = baseFolder.replace("\\", '/')
dark_home = darknet_home.replace("\\", '/')
cfgFolder = cfgFolder.replace("\\", '/')

darknet_path = os.path.join(dark_home, 'darknet')
cmd_file = os.path.join(cfgFolder, 'anchors.txt')

with open(cmd_file, 'w') as file:
    for i, size in enumerate(sizes):
        pline = "============================================================================================================="
        print(pline)
        file.write(pline+'\n')

        pline = "[{}x{}, {} boxes]".format(\
            sizes[i][1], sizes[i][1], sizes[i][0])

        print(pline)
        file.write(pline+'\n')
        print([darknet_path, 'detector', 'calc_anchors', os.path.join(cfgFolder, 'obj.data'), '-num_of_clusters', \
            str(sizes[i][0]), '-width', str(sizes[i][1]), '-height', str(sizes[i][1])])
        nmap_out = subprocess.run([darknet_path, 'detector', 'calc_anchors', os.path.join(cfgFolder, 'obj.data'), '-num_of_clusters', \
            str(sizes[i][0]), '-width', str(sizes[i][1]), '-height', str(sizes[i][1])], universal_newlines=False, stdout=subprocess.PIPE)

        nmap_lines = nmap_out.stdout.splitlines()
        ans = ''
        for line in nmap_lines:
            if 'anchors =' in str(line):
                ans = line.decode('ascii')
                print(ans)

        file.write(ans + '\n\n')


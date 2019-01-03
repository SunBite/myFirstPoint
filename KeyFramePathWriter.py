# -*- coding: utf-8 -*-
classname = ["basketball", "biking", "diving", "golf_swing", "horse_riding", "soccer_juggling","swing", "tennis_swing", "trampoline_jumping", "volleyball_spiking", "walking"]
with open("/home/sunbite/MFSSEL/keyframepath.txt",'a') as f:
    for i in classname:
        libsvmfilepath = "/home/sunbite/MFSSEL/keyframe/"+ i+" "
        f.writelines(libsvmfilepath)
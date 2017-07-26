#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:19:36 2017

@author: safy
"""

import tarfile
import os
import shutil
direc='/home/safy/Downloads/ILSVRC2014_DET_train/DataSet_5Classes/'
raw_Data='/home/safy/Downloads/ILSVRC2014_DET_train/'

for d in os.listdir(direc):
    if not d.endswith('.txt'):
        continue
    os.mkdir(direc+d.replace('.txt',''))
    os.chdir(direc)
    Data_Set=open(d,'r').readlines()
    for img in Data_Set:
        print (img)
        img=img.split('/')
        tar_folder=img[0]
        img_file=img[1]
        img_file=img_file[0:len(img_file)-1]
        folder=tarfile.open(raw_Data+tar_folder+'.tar')
        #Extracted_image = folder.extractfile(tar_folder+'/'+img_file+'.JPEG').read()
        folder.extractall(direc)
        
        
        shutil.copyfile(direc+'/'+tar_folder+'/'+img_file+'.JPEG',direc+d.replace('.txt','')+'/'+img_file)
        
        
       
        

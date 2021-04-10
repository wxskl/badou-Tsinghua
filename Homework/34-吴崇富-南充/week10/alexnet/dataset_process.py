#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
photos = os.listdir('./train')

with open('./dataset.txt','w') as f:
    for photo in photos:
        name = photo.split('.')[0]
        if name == 'cat':
            f.write(photo+';0\n')
        elif name == 'dog':
            f.write(photo+';1\n')
f.close()

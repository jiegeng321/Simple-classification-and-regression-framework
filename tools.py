#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
import os
import shutil
def is_img(img_name):
    return True if str(img_name).split(".")[-1].lower() in ["jpg","png","jpeg","gif"] and str(img_name)[0] != "." else False


def check_dir(path,delete=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if delete:
            shutil.rmtree(path)
            os.makedirs(path)

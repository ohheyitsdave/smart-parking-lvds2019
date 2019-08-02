import os
import numpy as np

def jpgs_and_labels(folder):
    filelist = []
    labels = []
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filen = os.path.join(root, filename)
            filelist.append(filen)
            if 'Empty' in filen:
                labels.append(0)
            else:
                 labels.append(1)
    return filelist, labels
"""source : https://github.com/jayrambhia/superpixels-SLIC/blob/master/SLICcv.py"""
import numpy as np


def connectivity(labels, step):
    h, w = labels.shape
    label = 0
    adjlabel = 0
    lims = int(w * h / step)
    dx4 = [-1, 0, 1, 0]
    dy4 = [0, -1, 0, 1]
    new_labels = -1 * np.ones_like(labels).astype(np.int64)
    elements = []
    for i in range(w):
        for j in range(h):
            if new_labels[j, i] == -1:
                elements = []
                elements.append((j, i))
                for dx, dy in zip(dx4, dy4):
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if 0 <= x < w and 0 <= y < h and new_labels[y, x] >= 0:
                        adjlabel = new_labels[y, x]
            count = 1
            c = 0
            while c < count:
                for dx, dy in zip(dx4, dy4):
                    x = elements[c][1] + dx
                    y = elements[c][0] + dy

                    if x >= 0 and x < w and y >= 0 and y < h:
                        if new_labels[y, x] == -1 and labels[j, i] == labels[y, x]:
                            elements.append((y, x))
                            new_labels[y, x] = label
                            count += 1
                c += 1
            if count <= lims >> 2:
                for c in range(count):
                    new_labels[elements[c]] = adjlabel
                label -= 1
            label += 1
    return new_labels

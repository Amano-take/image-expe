import numpy as np

class mixup():
    def __init__(self):
        pass

    def mixup(images, answers):
        B, len, _ = images
        halfB = B // 2
        
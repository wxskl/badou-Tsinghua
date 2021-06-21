import os

photos = os.listdir("F:/EVplayer/vidio/第十周/代码/alexnet/train/")

with open("data/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()

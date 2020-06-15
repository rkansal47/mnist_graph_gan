import imageio
from os import listdir

project_dir = "mnist_superpixels/"

figs_dir = project_dir + "figs/"
anim_dir = project_dir + "animations/"

dir_names = sorted(listdir(figs_dir))
dir_names = dir_names[41:43] + dir_names[44:49]

dir_names

full_dir = figs_dir + dir_names[0] +'/'
filenames = sorted(listdir(full_dir))

filenames

dir_names = ["56_lsgan_no_gru"]

for dir in dir_names:
    print(dir)
    name = anim_dir + dir
    full_dir = figs_dir + dir +'/'
    filenames = sorted(listdir(full_dir))

    with imageio.get_writer(name + ".gif", mode='I') as writer:
        i = 0
        for filename in filenames:
            # print(filename)
            if(i%20 == 0):
                print(i)
                try:
                    image = imageio.imread(full_dir + str(i) + ".png")
                    writer.append_data(image)
                except:
                    print("no 0.png")
            i += 5

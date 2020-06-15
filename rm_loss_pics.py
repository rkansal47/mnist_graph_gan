from os import listdir
from os import remove

project_dir = "mnist_superpixels/"

losses_dir = project_dir + "losses/"

# full_dir = losses_dir + "53_lsgan_num_layers_3/"
#
# pic_names = listdir(full_dir)
# pic_names[0].split('.')
# nums = []
# for pic in pic_names:
#     nums.append(int(pic.split('.')[0]))
#
# nums = sorted(nums)[:-1]
#
# nums
#
# for num in nums:
#     remove(full_dir + str(num) + '.png')
#

dir_names = sorted(listdir(losses_dir))[2:]
dir_names

for dir in dir_names:
    print(dir)
    full_dir = losses_dir + dir +'/'

    pic_names = listdir(full_dir)
    try:
        pic_names[0].split('.')
    except:
        continue
    nums = []
    for pic in pic_names:
        nums.append(int(pic.split('.')[0]))

    nums = sorted(nums)[:-1]

    for num in nums:
        remove(full_dir + str(num) + '.png')

import os

path = f"obstacle-detection/images"
filelist_dict = { ind: name for (ind,name) in enumerate([files for files in os.listdir(f"{path}/obstacle")]) }
directory = f"{path}/obstacle"
frames = 3
# num_files = count_files_in_directory(directory)


for i in filelist_dict:
    image_name = int(filelist_dict[i][:-4])
    for j in range(1, frames):
        new_img = f"{image_name - j}.png"
        print(new_img)
        os.replace(f"{path}/no_obstacle/{new_img}", f"{path}/obstacle/{new_img}")
import os


dir = os.getcwd()

for d in os.listdir():
    if "Data" in d:
        os.chdir(d)

for d in os.listdir():
    if "head" in d:
        os.chdir(d)

data_label = os.listdir()


tot_dir = [0, 0]
list_of_IDs = []

for label in data_label:
    fish_ID, direction, image_nr = label.split('_')
    image_nr, _ = image_nr.split('.')

    if direction == 'l':
        tot_dir[0] += 1
    else:
        tot_dir[1] += 1

    if not fish_ID in list_of_IDs:
        list_of_IDs.append(fish_ID)


print("Total number of fish IDs:", len(list_of_IDs))
print("Number of left images:", tot_dir[0])
print("Number of right images:", tot_dir[1])


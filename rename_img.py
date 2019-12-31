import os 

# don't add "/" to the end of DATA_DIRECTORY
DATA_DIRECTORY = "./JPP_mod/output/parsing/val"

def main():
    print("Loading images...\n")
    idList= []
    for image in os.listdir(DATA_DIRECTORY):
        image_name = image[:-4]
        try:
            if image_name[4] == "_":
                original_id = image_name.split("_")
                
                if not original_id[0] in idList:
                    idList.append(original_id[0])
        except IndexError:
            continue

    idList.sort()
    nameList = []
    renameTable = {}
    for i, id in enumerate(idList):
        name = str(i)
        while len(name) < 4:
            name = "0" + name
        renameTable[id] = name

    print("Start renaming...\n")
    cnt = 0
    for image in os.listdir(DATA_DIRECTORY):
        image_name = image[:-4]
        file_type = image[-4:]
        
        if image_name[4] == "_":
            original_id = image_name.split("_")
            newName = "/" + renameTable[original_id[0]] + "_" + original_id[1] + file_type
            
            try:
                os.rename(DATA_DIRECTORY+'/'+image, DATA_DIRECTORY+newName)
            except:
                print(f"error occurs when renaming {DATA_DIRECTORY+'/'+image} !")
        cnt += 1
        if cnt % 1000 == 0:
            print(f"Already converted {cnt} images...\n")

    print(f"All {cnt} images converted!\n")

if __name__ == "__main__":
    main()

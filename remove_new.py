import os 

# don't add "/" to the end of DATA_DIRECTORY
DATA_DIRECTORY = "./JPP_mod/output/parsing/val"

def main():
    print("Start removing _new...\n")

    cnt = 0
    for image in os.listdir(DATA_DIRECTORY):
        image_name = image[:-4]
        file_type = image[-4:]
        
        try:
            os.rename(DATA_DIRECTORY+'/'+image, DATA_DIRECTORY+"/"+image_name[:-4]+file_type)
        except:
            print(f"error occurs when renaming {DATA_DIRECTORY+'/'+image} !")
        cnt += 1

    print(f"All {cnt} images converted!\n")

if __name__ == "__main__":
    main()

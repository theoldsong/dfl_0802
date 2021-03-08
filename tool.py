import os
import shutil
path="E:\DFL"

def get_filelist(dir):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in dirs:
            Filelist.append(os.path.join(home, filename))
    
    return Filelist

if __name__ =="__main__":
    Filelist = get_filelist(dir)
    print(len( Filelist))
    for file in Filelist :
        if "__pycache__" in file:   
            if os.path.exists(file):
                print(file)
                shutil.rmtree(file)

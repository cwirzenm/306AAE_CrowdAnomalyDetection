import os

a = open("Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/output.txt", "a")
for path, subdirs, files in os.walk(r'Y:/FINAL PROJECT/UCF_Crime/UCF_Crimes/Videos/New folder'):
    for filename in files:
        f = os.path.join(path.split("\\")[1], filename)
        a.write(str(f) + "\n")

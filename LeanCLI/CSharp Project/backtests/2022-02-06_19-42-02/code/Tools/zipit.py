from zipfile import ZipFile
import os

files = os.listdir('.')
print(files)
for file in files:
  # do something
    filename = file[:-4]
    filename = filename + ".zip"
    with ZipFile(filename, 'w') as file:
        print("{} is created.".format(filename))
        filename = [:-4]
        filename = filename + ".csv"
        file.write(filename)

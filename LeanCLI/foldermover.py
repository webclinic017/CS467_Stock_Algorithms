
# Python program to move
# files and directories
  
  
import shutil
import time
import os

# Source path

while True:
    time.sleep(5)
    source = r"C:\Users\Hats\Documents\School\CS467\Project Files\CS467_Stock_Algorithms\LeanCLI\CSharp Project\backtests\\"
    folder = ""
    if len(os.listdir(source)) > 10:
        for files in os.listdir(source):
            folder = files
            source = source + str(files)
            break
        # Destination path
        destination = r"U:\Backtests"
          
        # Move the content of
        # source to destination
        print("Moving: ", folder)
        dest = shutil.move(source, destination)
      



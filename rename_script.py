# Pythono3 code to rename multiple 
# files in a directory or folder
  
# importing os module
import os
  
# Function to rename multiple files
def main():
    REPLACE_STR = "_edges"
    for count, filename in enumerate(os.listdir("./")):
        src = dst = filename
        if REPLACE_STR in filename:
            dst = filename.replace(REPLACE_STR, "")
        # rename() function will
        # rename all the files
        os.rename(src, dst)
  
# Driver Code
if __name__ == '__main__':
      
    # Calling main() function
    main()
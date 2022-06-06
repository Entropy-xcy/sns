from synpred import fs, fs_files
import sys

def delete_filename_one(filename):
    print(filename)
    fid = fs_files.find_one({"filename": filename})['_id']
    fs.delete(fid)
    print("Done")

if __name__ == "__main__":
    for i in range(1, 2):
        delete_filename_one("{}.gexf".format(i))

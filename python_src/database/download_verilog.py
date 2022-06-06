import sys
import base64
from synpred import infer_top_name, infer_clock_name, download_design, file2b64
import hashlib

did = int(sys.argv[1])
des = download_design(did)
filename = des['top_name']+".v"
outfile = open(filename, "wb")
outfile.write(des['source'])
outfile.close()

# Verify Hash
filehash = hashlib.md5(des['source']).hexdigest()
db_hash = des['hash']
try: 
    assert filehash == db_hash
except:
    raise RuntimeError("Inconsistent Hash, File Corrupted.\n Please Retry.")

print("did={}".format(did))
print("Done!")

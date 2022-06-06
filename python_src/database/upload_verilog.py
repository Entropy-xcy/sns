import sys
import base64
from synpred import infer_top_name, infer_clock_name, upload_design, get_did_from_hash
import hashlib

filename = sys.argv[1]


description = ""
if len(sys.argv) > 2:
    description = " ".join(sys.argv[2:])

sfile = open(filename, "rb")
source = sfile.read()
top_name = infer_top_name(filename)
clock_name = infer_clock_name(filename)
design_hash = hashlib.md5(source).hexdigest()
print("Uploading Design:")
print("\tTop: {}\tClock: {}".format(top_name, clock_name))
print("\tHash: {}".format(design_hash))
print("\tDescription: {}".format(description))
try: 
    upload_design(top_name, clock_name, source, description=description)
    print("Done!")
    did = get_did_from_hash(design_hash)
    print("did={}".format(did))
except:
    print("Duplicated Design, Aborted!")
    did = get_did_from_hash(design_hash)
    print("did={}".format(did))
finally:
    sfile.close()

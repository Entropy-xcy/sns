import pymongo
import base64
import hashlib
import pytz
from datetime import datetime
import gridfs
import networkx as nx
from networkx.readwrite.gexf import write_gexf, read_gexf
import os
# timezone = pytz.timezone('US/Pacific')

#########################################################
# Credential Information below is removed for open-source
#########################################################
DB_USER = ""
DB_PWD = ""
DB_HOST = ""
DB_NAME = ""
RAW_GRAPH_PIPE = ""

url = "mongodb://{}:{}@{}/?authSource={}".format(DB_USER, DB_PWD, DB_HOST, DB_NAME)
mclient = pymongo.MongoClient(url)

# Main Database
mdb = mclient[DB_NAME]

# GridFS
fs = gridfs.GridFS(mdb)

# Collections
design = mdb["design"]
ir     = mdb["ir"]
asic   = mdb["asic"]
fpga   = mdb["fpga"]
fs_files = mdb["fs.files"]

jtype_list = ["ir", "asic", "fpga"]

ir_stype_list = ["count", "graph", "raw_graph", "ud_nf_graph", "seq_gen"]
fpga_stype_list = ["virtex_us", "stratix"]
asic_stype_list = ["pdk15", "syno45"]

stype_list = {
                "ir": ir_stype_list,
                "fpga": fpga_stype_list,
                "asic": asic_stype_list
             }


def file2b64(filename):
    dsrc = open(filename, "rb").read()
    return base64.b64encode(dsrc)

def b642file(b64, filename):
    dsrc = open(filename, "wb")
    dsrc.write(base64.b64decode(b64))
    dsrc.close()

# Upload the Design (Verilog Source File) to the Database
def upload_design(top_name, clock_name, source, 
                    name="", description="", esti_mem=0.0, dtype="verilog"):
    did = 1
    if design.count_documents({}) > 0:
        dids = design.find().sort("did", -1).limit(1)
        did = dids[0]['did'] + 1
    design_hash = hashlib.md5(source).hexdigest()
    upload_time = pytz.utc.localize(datetime.now())

    src_filename = top_name+".v"
    # Upload source to FS
    file_id = fs.put(source, filename=src_filename)

    # upload_time = timezone.localize(datetime.now())
    if design.count_documents({"hash": design_hash}) == 0:
        design.insert_one({
                        "did": did, 
                        "dtype": dtype, 
                        "name": name, 
                        "description": description,
                        "source": file_id,
                        "esti_mem": esti_mem,
                        "top_name": top_name,
                        "clock_name": clock_name,
                        "hash": design_hash,
                        "date": upload_time
                    })
    else:
        raise RuntimeError("Duplicate Design Found")

# Return -1 for design with hash does not exist
def get_did_from_hash(hash):
    if design.count_documents({"hash": hash}) > 0:
        return design.find({"hash": hash}).limit(1)[0]['did']
    else:
        return -1

# Download the Design with did from the database.
def download_design(did):
    count = design.count_documents({"did": did})
    if count > 1:
        raise RuntimeError("Duplicated Design, Database Corrupted!")
    if count <= 0:
        raise RuntimeError("Design with did={} does not exist.".format(did))
    dsrc = design.find({"did": did}).limit(1)[0]
    source_binary = fs.get(dsrc['source']).read()
    dsrc['source'] = source_binary
    return dsrc

# Get the did of next job. And set the next job to running at the same time.
# If there is no job in the queue, return -1
def get_next_job(jtype, stype):
    if jtype not in ['ir', 'asic', 'fpga']:
        raise RuntimeError("jtype Need to be \'ir\', \'asic\' or \'fpga\'.")
    table = mdb[jtype]
    if table.count_documents({'status': 'new', 'stype': stype}) > 0:
        todo = table.find({'status': 'new', 'stype': stype}).limit(1)[0]
        table.update_one(todo, {"$set": {'status': 'running'}})
        return todo['did']
    else:
        return -1

# Set the job status of stype did.
def set_job_status(jtype, did, stype, status):
    if jtype not in ['ir', 'asic', 'fpga']:
        raise RuntimeError("jtype Need to be \'ir\', \'asic\' or \'fpga\'.")
    table = mdb[jtype]

    if table.count_documents({'did': did, 'stype': stype}) > 0:
        # Entry with did and stype exist then update it
        table.update_one({'did': did, 'stype': stype}, {"$set": {'status': status}})
    else:
        raise RuntimeError("Entry does not exist")

# Get the job status of stype did.
def get_job_status(jtype, did, stype):
    if jtype not in ['ir', 'asic', 'fpga']:
        raise RuntimeError("stype Need to be \'ir\', \'asic\' or \'fpga\'.")
    table = mdb[jtype]
    
    if table.count_documents({'did': did, 'stype': stype}) > 0:
        return table.find({'did': did, 'stype': stype}).limit(1)[0]['status']
    else:
        raise RuntimeError("Entry does not exist")

def init_fpga_job(did, stype):
    if fpga.count_documents({'did': did, "stype": stype}) > 0:
        raise RuntimeError("Entry did={} already exist in collection fpga, cannot init again.".format(did))
    fpga.insert_one({
        'did': did,
        'status': "new",
        'stype': stype,
        'luts': 0,
        'dsps': 0,
        'ffs': 0,
        'power': 0.0,
        'timing': 0.0,
        'reports': b""
    })

def init_asic_job(did, stype):
    if asic.count_documents({'did': did, "stype": stype}) > 0:
        raise RuntimeError("Entry did={} already exist in collection asic, cannot init again.".format(did))
    asic.insert_one({
        'did': did,
        'status': "new",
        'stype': stype,
        'area': 0.0,
        'power': 0.0,
        'timing': 0.0,
        'reports': b""
    })

def init_ir_job(did, stype):
    if ir.count_documents({'did': did, "stype": stype}) > 0:
        raise RuntimeError("Entry did={} already exist in collection ir, cannot init again.".format(did))
    ir.insert_one({
        'did': did,
        'status': "new",
        'stype': stype,
        'result': b"",
        'reports': b""
    })

# Initialize Job of jtype and stype
def init_job(did, jtype, stype):
    assert jtype in jtype_list
    if jtype == "ir":
        init_ir_job(did, stype)
    elif jtype == "fpga":
        init_fpga_job(did, stype)
    elif jtype == "asic":
        init_asic_job(did, stype)

# Upload Result of jtype and stype to the Database
# Set the status of the job to be "done"
def upload_result(did, jtype, stype, results):
    assert jtype in jtype_list
    assert stype in stype_list[jtype]
    if mdb[jtype].count_documents({'did': did, 'stype': stype}) <= 0:
        raise RuntimeError("Result Entry have to be Init first before upload.")
    mdb[jtype].update_one({'did': did, 'stype': stype}, {'$set': results})
    set_job_status(jtype, did, stype, 'done')

def delete_result(did, jtype, stype):
    assert jtype in jtype_list
    assert stype in stype_list[jtype]
    if mdb[jtype].count_documents({'did': did, 'stype': stype}) <= 0:
        raise RuntimeError("No result with type ({}, {}) found. Nothing to delete".format(jtype, stype))
    mdb[jtype].delete_one({'did': did, 'stype': stype})

# Delete all the entries with did in all the collections
def delete_did(did):
    fileid = design.find_one({"did": did})['source']
    fs.delete(fileid)
    design.remove({"did": did})
    asic.remove({"did": did})
    fpga.remove({"did": did})
    ir.remove({"did": did})

if __name__ == "__main__":
    # print(get_next_job('ir', 'yosys'))
    a = fs.put(b"5fdffdae71dc30636ae52139")
    print(a)
    b = fs.get(a)
    print(b.read())


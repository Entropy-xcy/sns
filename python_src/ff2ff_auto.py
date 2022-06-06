import sys
import base64
from synpred import infer_top_name, infer_clock_name, upload_design, get_did_from_hash, upload_result, init_job
from syno_synthesis_did import syno_synthesis_did, report_error
import hashlib


def ff2ff_auto(vfile, seqfile):
    vfile = sys.argv[1]
    seqfile = sys.argv[2]
    source = open(vfile, "rb").read()
    top_name = infer_top_name(vfile)
    clock_name = infer_clock_name(vfile)
    design_hash = hashlib.md5(source).hexdigest()

    # Upload Verilog First
    try:
        upload_design(top_name, clock_name, source, description="ff2ff path dataset")
    except RuntimeError:
        # Skip on duplicated designs
        print("Duplicated Design Found, skip")
        return
    except:
        raise RuntimeError("Fatal Error, Terminate")

    did = get_did_from_hash(design_hash)
    print("Did = {}".format(did))

    # Synthesis Did
    try:
        print("Synthesising for did={}".format(did))
        syno_synthesis_did(did, "pdk15")
    except:
        report_error(did, "pdk15")

    # Upload sequence to IR
    init_job(did, "ir", "seq_gen")
    seq = open(seqfile, "r").read().replace(" ", "").split("\n")
    while ("" in seq):
        seq.remove("")
    results = {"result": seq}
    upload_result(did, "ir", "seq_gen", results)
    print("Seq Result Uploaded!")


if __name__ == "__main__":
    vfile = sys.argv[1]
    seqfile = sys.argv[2]
    ff2ff_auto(vfile, seqfile)

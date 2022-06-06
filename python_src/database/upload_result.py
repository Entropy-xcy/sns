import sys
from synpred import upload_result, init_job, download_design, jtype_list, stype_list

def input_asic_result():
    area = float(input("Area (mm^2) of the Design? -> "))
    power = float(input("Power (mW) of the Design? -> "))
    timing = float(input("Critical Delay (ps) of the Design? -> "))
    rpt_path = input("Path for Reports Tarball? (Empty for no report) -> ")
    reports = b''
    if rpt_path != "":
        reports = open(rpt_path, "rb").read()
    asic_result = {
        "area": area,
        "power": power,
        "timing": timing,
        'reports': reports
    }
    return asic_result

def input_fpga_result():
    luts = int(input("#LUTs of the Design? -> "))
    dsps = int(input("#DSPs of the Design? -> "))
    ffs = int(input("#FFs of the Design? -> "))
    power = float(input("Power (mW) of the Design? -> "))
    timing = float(input("Critical Delay (ps) of the Design? -> "))
    rpt_path = input("Path for Reports Tarball? (Empty for no report) -> ")
    reports = b''
    if rpt_path != "":
        reports = open(rpt_path, "rb").read()
    fpga_result = {
        'luts': luts,
        'dsps': dsps,
        'ffs': ffs,
        'power': power,
        'timing': timing,
        'reports': reports
    }
    return fpga_result


did = int(sys.argv[1])
des = download_design(did)
description = des["description"]
print("Entering Result for did={}".format(did))
print("Description: {}".format(description))

jtype = input("Which Flow? [\"ir\", \"asic\", \"fpga\"] -> ")
assert jtype in jtype_list
stype = input("Synthesis Type? {} -> ".format(str(stype_list[jtype])))

result = {}
if jtype == 'asic':
    result = input_asic_result()
elif jtype == 'fpga':
    result = input_fpga_result()
else:
    raise RuntimeError("IR is not supported yet!")

init_job(did, jtype, stype)
upload_result(did, jtype, stype, result)

print("Uploaded!")

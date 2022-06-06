from syno_synthesis_did import read_syno_results
from synpred import upload_result, init_job, set_job_status
import sys

if __name__ == "__main__":
    did = int(sys.argv[1])
    stype = sys.argv[2]
    job_path = sys.argv[3]
    try:
        init_job(did, "asic", stype)
    except:
        print("ASIC Entry already exist")
    area, power, timing, report = read_syno_results(job_path)
    dpower = power[0]
    lpower = power[1]
    power = dpower + lpower
    asic_result = {
        "area": area,
        "power": power,
        "dpower": dpower,
        "lpower": lpower,
        "timing": timing,
        'reports': report
    }

    # Upload Results to Database
    upload_result(did, "asic", stype, asic_result)

import subprocess
import time
import os
import _thread
from synpred import download_design, init_job, upload_result, set_job_status
import sys
VVD_BIN = "/tools/Xilinx/Vivado/2020.2/bin/vivado"+" "+"-mode"+" "+"tcl"
VVD_LIB = {
    "HOME":"/home/yujie"
}
#VVD_LIB = {
#    "LD_LIBRARY_PATH":"/home/home5/cx60/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
#    "SNPSLMD_LICENSE_FILE":"27060@license01.egr.duke.edu",
#    "HOME":"/home/home5/cx60"
#}
ORIG_TIMING = 10

def monitor_stdout(process):
    while True:
        try:
            print(process.stdout.readline().decode(), end="")
        except:
            return

def getcwd(relative_path=""):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)

def proc_luts(summary_report):
    begin_idx = summary_report.find("LUT as Logic") + 38
    luts_report = summary_report[begin_idx:]
    end_idx = luts_report.find("|")
    selected = luts_report[:end_idx]
    return float(selected)

def proc_dsps(summary_report):
    begin_idx = summary_report.find("DSPs") + 40
    dsps_report = summary_report[begin_idx:]
    end_idx = dsps_report.find("|")
    selected = dsps_report[:end_idx]
    return float(selected)

def proc_ffs(summary_report):
    begin_idx = summary_report.find("F7/F8 Muxes") + 40
    ffs_report = summary_report[begin_idx:]
    end_idx = ffs_report.find("|")
    selected = ffs_report[:end_idx]
    return float(selected)

def proc_dpower(summary_report):
    begin_idx = summary_report.find("Dynamic (W)")  + 27
    power_report = summary_report[begin_idx:]
    end_idx = power_report.find("|")
    selected = power_report[:end_idx]
    return 1000*float(selected)

def proc_spower(summary_report):
    begin_idx = summary_report.find("Static Power")  + 25
    power_report = summary_report[begin_idx:]
    end_idx = power_report.find("|")
    selected = power_report[:end_idx]
    return 1000*float(selected)

def proc_timing(summary_report):
    # delay = required time - slack time
    met = True
    if "inf" in summary_report:
        met = False
    # find slack time
    begin_idx = summary_report.find("Slack (MET) :") + 25
    slack_report = summary_report[begin_idx:]
    end_idx = slack_report.find("ns")
    slack = slack_report[:end_idx]
    # find required time
    begin_idx = summary_report.find("Requirement: ") + 22
    req_report = summary_report[begin_idx:]
    end_idx = req_report.find("ns")
    req = req_report[:end_idx]

    return abs(1000*(float(req) - float(slack))), met


def read_vvd_results(rel_path, workdir="work/", outdir="output/"):
    
    outfile_list = os.listdir(os.path.join(rel_path, outdir))             
    summary_report_filename = ""
    for f in outfile_list:
        if "summary" in f:
            summary_report_filename = f
        
    summary_report_filename = os.path.join(rel_path, outdir, summary_report_filename)
    summary_report = open(summary_report_filename, "r").read()
    luts = proc_luts(summary_report)
    dsps = proc_dsps(summary_report)
    ffs = proc_ffs(summary_report)
    dpower = proc_dpower(summary_report)
    spower = proc_spower(summary_report)
    timing = proc_timing(summary_report)
    return luts, dsps, ffs, dpower, spower, timing

def vvd_synthesis(source, topname, clockname, rel_path="synthesis/",
                        library="syn_libraries/nanGate_15_CCS_typical.db",
                        workdir="/work/", outdir="output", orig_timing=ORIG_TIMING):
    cwd = getcwd()
    environment = {
        "projdir":cwd,
        "reldir":rel_path,
        "outdir":outdir,
        "workdir":workdir,
        "library":library,
        "source":source,
        "topname":topname,
        "clockname":clockname,
        "orig_timing": str(orig_timing)
    }
    print(os.path.join(rel_path,outdir))
    os.mkdir(os.path.join(rel_path, outdir))
    environment.update(VVD_LIB)
    vvd = subprocess.Popen([VVD_BIN], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True, env=environment)
    _thread.start_new_thread(monitor_stdout, (vvd, ))
    vvd.communicate(input=b"source ../tcl_src/fpga_syn.tcl")
    print("Done!")


def vvd_synthesis_did(did, stype, syn_path="synthesis/", auto_syn=True):
    if stype != "virtex_us":
        raise RunTimeError("Synthesis Other than virtex_us is not supported Yet!")
    dsrc = download_design(did)
    # Job Name Format: <did>_<jtype>_<stype>
    job_name = "{}_{}_{}".format(str(did), "fpga", stype)
    job_path = os.path.join(syn_path, job_name)
    
    # Create Directory if not exist
    try:
        os.mkdir(job_path)
    except:
        pass
    
    # Preparing for synthesis
    topname = dsrc['top_name']
    filename = topname + ".v"
    clockname = dsrc['clock_name']
    outfile = open(os.path.join(job_path, filename), "wb")
    outfile.write(dsrc['source'])
    outfile.close()

    # Init Job
    init_job(did, "fpga", stype)
    set_job_status("fpga", did, stype, "running")

    global_start = time.time()

    if auto_syn:
        # Start Synthesis
        start = time.time()
        vvd_synthesis(filename, topname, clockname, rel_path=job_path)
        end = time.time()
        time_elapsed = end - start
        total_time = end - global_start

        # Read Synthesis Results
        luts, dsps, ffs, dpower, spower, timing_tuple = read_vvd_results(job_path)
        timing = timing_tuple[0]
        met = timing_tuple[1]
        fpga_result = {
            "luts": luts,
            "dsps": dsps,
            "ffs": ffs,
            "dynamic power": dpower,
            "static power": spower,
            "timing": timing,
           # 'reports': report,
            "runtime": time_elapsed,
            "totaltime": total_time,
            "slack_met": met,
        }

        # Upload Results to Database
        upload_result(did, "fpga", stype, fpga_result)

if __name__ == "__main__":
    # syno_synthesis_did("XmcmcDecoderMT.v", "XmcmcDecoderMT", "clock")
    did = int(sys.argv[1])
    vvd_synthesis_did(did, "virtex_us")
    # print(tar_reports("synthesis/1_asic_pdk15")[:100])
    # area, power, timing, report = read_syno_results("synthesis/6_asic_pdk15/")
    # print(area, power, timing)

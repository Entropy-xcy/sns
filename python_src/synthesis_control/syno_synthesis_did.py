import subprocess
import time
import os
import _thread
from synpred import download_design, init_job, upload_result, set_job_status, get_job_status
import sys

DC_BIN = "/usr/pkg/synopsys-5.0.1/syn/Q-2019.12-SP4/bin/dc_shell"
DC_LIB = {
    "LD_LIBRARY_PATH":"/home/home5/cx60/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
    "SNPSLMD_LICENSE_FILE":"27060@license01.egr.duke.edu",
    "HOME":"/home/home5/cx60"
}

TIMING_STEP_CONST = 0.95
ORIG_FREQ = 2800

def monitor_stdout(process):
    while True:
        try:
            print(process.stdout.readline().decode(), end="")
        except:
            return


def getcwd(relative_path=""):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)


def proc_area(area_report):
    begin_idx = area_report.find("Total cell area:") + len("Total cell area:") + 1
    area_report = area_report[begin_idx:]
    end_idx = area_report.find("\n")
    selected = area_report[:end_idx]
    return float(selected)


def proc_power(power_report):
    begin_idx = power_report.find("Total Dynamic Power") + len("Total Dynamic Power")
    power_report = power_report[begin_idx:]
    begin_idx = power_report.find("=") + len("=")
    power_report = power_report[begin_idx:]

    dpower_mW = power_report[: power_report.find("mW")]
    dpower_W = power_report[: power_report.find("W")]
    dpower_uW = power_report[: power_report.find("uW")]
    dpower = 0.0
    try: 
        dpower = float(dpower_mW)
    except:
        try:
            dpower = float(dpower_W) * 1000.0
        except:
            try:
                dpower = float(dpower_uW) / 1000.0
            except:
                pass

    begin_idx = power_report.find("Cell Leakage Power") + len("Cell Leakage Power")
    power_report = power_report[begin_idx:]
    begin_idx = power_report.find("=") + len("=")
    power_report = power_report[begin_idx:]

    lpower_mW = power_report[: power_report.find("mW")]
    lpower_W = power_report[: power_report.find("W")]
    lpower_uW = power_report[: power_report.find("uW")]
    lpower = 0.0
    try: 
        lpower = float(lpower_mW)
    except:
        try:
            lpower = float(lpower_W) * 1000.0
        except:
            try:
                lpower = float(lpower_uW) / 1000.0
            except:
                pass

    return dpower, lpower

def proc_timing(timing_report):
    met = False
    if "slack (MET)" in timing_report:
        met = True
    begin_idx = timing_report.find("data arrival time") + len("data arrival time") + 1
    timing_report = timing_report[begin_idx:]
    end_idx = timing_report.find("\n")
    selected = timing_report[:end_idx]

    return abs(float(selected)), met


def tar_reports(rel_path, workdir="work/", outdir="out/"):
    tar_proc = subprocess.Popen(["tar", "cvf", rel_path+".tar.gz", os.path.join(rel_path, outdir)])
    tar_proc.wait()
    tarfile = open(rel_path+".tar.gz", "rb")
    tarbin = tarfile.read()
    tarfile.close()
    # os.remove(rel_path+".tar.gz")
    return tarbin


def read_syno_results(rel_path, workdir="work/", outdir="out/"):
    outfile_list = os.listdir(os.path.join(rel_path, outdir))
    area_report_filename = ""
    power_report_filename = ""
    timing_report_filename = ""
    for f in outfile_list:
        if "area" in f:
            area_report_filename = f
        elif "power" in f:
            power_report_filename = f
        elif "timing" in f:
            timing_report_filename = f
        elif "netlist" in f:
            netlist_report_filename = os.path.join(rel_path, outdir, f)
            os.remove(netlist_report_filename)
    area_report_filename = os.path.join(rel_path, outdir, area_report_filename)
    power_report_filename = os.path.join(rel_path, outdir, power_report_filename)
    timing_report_filename = os.path.join(rel_path, outdir, timing_report_filename)
    area_report = open(area_report_filename, "r").read()
    power_report = open(power_report_filename, "r").read()
    timing_report = open(timing_report_filename, "r").read()
    area = proc_area(area_report)
    power = proc_power(power_report)
    timing = proc_timing(timing_report)
    report = tar_reports(rel_path)
    return area, power, timing, report


def syno_synthesis(source, topname, clockname, rel_path="synthesis/",
                        library="syn_libraries/nanGate_15_CCS_typical.db",
                        workdir="/work/", outdir="/out/", target_freq=ORIG_FREQ):
    cwd = getcwd()
    environment = {
        "projdir":cwd+"../",
        "reldir":rel_path,
        "outdir":outdir,
        "workdir":workdir,
        "library":library,
        "source":source,
        "topname":topname,
        "clockname":clockname,
        "freq": str(target_freq)
    }
    environment.update(DC_LIB)
    dc = subprocess.Popen([DC_BIN], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True, env=environment)
    _thread.start_new_thread(monitor_stdout, (dc, ))
    dc.communicate(input=b"source tcl_src/asic_syn.tcl")
    print("Done!")


def syno_synthesis_did(did, stype, syn_path="synthesis/", auto_syn=True):
    if stype != "pdk15":
        raise RunTimeError("Synthesis Other than PDK15 is not supported Yet!")
    dsrc = download_design(did)
    # Job Name Format: <did>_<jtype>_<stype>
    job_name = "{}_{}_{}".format(str(did), "asic", stype)
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
    init_job(did, "asic", stype)
    set_job_status("asic", did, stype, "running")

    global_start = time.time()

    if auto_syn:
        met = False
        freq = ORIG_FREQ
        iter = 1
        while not met:
            # Start Synthesis
            start = time.time()
            syno_synthesis(filename, topname, clockname, rel_path=job_path, target_freq=freq)
            end = time.time()
            time_elapsed = end - start
            total_time = end - global_start

            # Read Synthesis Results
            area, power, timing_tuple, report = read_syno_results(job_path)
            dpower = power[0]
            lpower = power[1]
            power = dpower + lpower
            timing = timing_tuple[0]
            met = timing_tuple[1]
            asic_result = {
                "area": area,
                "power": power,
                "dpower": dpower,
                "lpower": lpower,
                "timing": timing,
                'reports': report,
                "runtime": time_elapsed,
                "totaltime": total_time,
                "slack_met": met,
                "iteration": iter
            }

            # Upload Results to Database
            upload_result(did, "asic", stype, asic_result)
            set_job_status("asic", did, stype, "running")
            freq = 1000*1000/timing * TIMING_STEP_CONST
            iter = iter + 1
            if iter >= 3:
                break
            print("New Freq: ", freq)
        set_job_status("asic", did, stype, "done")
    else:
        # Start Synthesis
        start = time.time()
        syno_synthesis(filename, topname, clockname, rel_path=job_path)
        end = time.time()
        time_elapsed = end - start
        total_time = end - global_start

        # Read Synthesis Results
        area, power, timing_tuple, report = read_syno_results(job_path)
        dpower = power[0]
        lpower = power[1]
        power = dpower + lpower
        timing = timing_tuple[0]
        met = timing_tuple[1]
        asic_result = {
            "area": area,
            "power": power,
            "dpower": dpower,
            "lpower": lpower,
            "timing": timing,
            'reports': report,
            "runtime": time_elapsed,
            "totaltime": total_time,
            "slack_met": met,
            "iteration": 1
        }

        # Upload Results to Database
        upload_result(did, "asic", stype, asic_result)

def report_error(did, stype):
    # Init Job
    try:
        init_job(did, "asic", stype)
    except:
        if get_job_status("asic", did, stype) == "done":
            return
    
    set_job_status("asic", did, stype, "error")

if __name__ == "__main__":
    # syno_synthesis_did("XmcmcDecoderMT.v", "XmcmcDecoderMT", "clock")
    did = 0
    try:
        did = int(sys.argv[1])
    except:
        raise ValueError("You have to specify did by e.g. \"make synopsys_synthesis did=1\"")
    
    try:
        print("Synthesising for did={}".format(did))
        syno_synthesis_did(did, "pdk15")
    except:
        report_error(did, "pdk15")

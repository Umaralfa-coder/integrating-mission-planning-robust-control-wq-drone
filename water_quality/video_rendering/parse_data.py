import csv, re, numpy as np, pickle

EPOCH_OFFSET = 1786312800.0
MISSION_CSV = "../../data/control_missions/2026-08-10_09-34-26_182_mine_RNLPIDmission_260810_091911.csv"
COOLTERM = "../../data/water_quality/coolterm_capture_2026-08-06_mission.txt"

def tod_to_sec(hhmmss):
    hhmmss = float(hhmmss)
    hh = int(hhmmss // 10000)
    mm = int((hhmmss % 10000) // 100)
    ss = hhmmss % 100
    return hh*3600 + mm*60 + ss

# ---- 1. parse mission CSV for boat position vs epoch ----
epochs, xs, ys, lats, lons, headings = [], [], [], [], [], []
with open(MISSION_CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        epochs.append(float(row['DEBUG_timestamp']))
        xs.append(float(row['DEBUG_pos_x_m']))
        ys.append(float(row['DEBUG_pos_y_m']))
        lats.append(float(row['latitude_deg']))
        lons.append(float(row['longitude_deg']))
        headings.append(float(row['heading_deg']))
epochs = np.array(epochs); xs = np.array(xs); ys = np.array(ys)
lats = np.array(lats); lons = np.array(lons); headings = np.array(headings)
print("CSV mission: N=", len(epochs), "epoch range", epochs[0], epochs[-1], "duration", epochs[-1]-epochs[0])

# ---- 2. parse CoolTerm log lines for date=100826, extract time, lat/lon, DO_mgL ----
pat = re.compile(r'^100826;(\d+\.\d+);([\-0-9.]+);([\-0-9.]+);([\-0-9.]+);([^;]*);([^;]*);([^;]*);([^;]*);?')

ct_epoch, ct_lat, ct_lon, ct_do, ct_ph, ct_temp, ct_cond, ct_orp, ct_turb = [],[],[],[],[],[],[],[],[]

def nmea_to_dd(v):
    if v == 0.0:
        return None
    deg = int(v/100)
    minutes = v - deg*100
    return deg + minutes/60.0

with open(COOLTERM, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line.startswith('100826;'):
            continue
        parts = line.split(';')
        if len(parts) < 9:
            continue
        try:
            t_field = parts[1]
            lat_raw = float(parts[2])
            lon_raw = float(parts[3])
            rs232 = parts[5]
            sdi1 = parts[6]
            sdi2 = parts[7]
            sdi3 = parts[8]
            tod = tod_to_sec(t_field)

            # SDI12_1 -> DO probe: addr,temp,pct_sat,do_mgl,do_mgl2,err
            s1 = sdi1.split(',')
            do_mgl = float(s1[3]) if len(s1) > 3 else None
            temp1 = float(s1[1]) if len(s1) > 1 else None

            # SDI12_2 -> pH/ORP/Cond probe: addr,temp,pH,concat(cond,orp),err
            s2 = sdi2.split(',')
            ph_val = float(s2[2]) if len(s2) > 2 else None
            cond_val = None
            orp_val = None
            if len(s2) > 3:
                m = re.match(r'^(-?\d+\.\d+)(-\d+\.\d+)$', s2[3])
                if m:
                    cond_val = float(m.group(1))/1000.0
                    orp_val = float(m.group(2))

            # RS232 turbidity: $PVHY2,80,8,6.428,,NTU,132.667,,NTU,M4,*63
            turb_val = None
            rm = re.search(r'\$PVHY2,\d+,\d+,([\-0-9.]+),', rs232)
            if rm:
                turb_val = float(rm.group(1))

            ct_epoch.append(EPOCH_OFFSET + tod)
            ct_lat.append(nmea_to_dd(lat_raw))
            ct_lon.append(nmea_to_dd(lon_raw))
            ct_do.append(do_mgl)
            ct_ph.append(ph_val)
            ct_temp.append(temp1)
            ct_cond.append(cond_val)
            ct_orp.append(orp_val)
            ct_turb.append(turb_val)
        except Exception as e:
            continue

ct_epoch = np.array(ct_epoch)
print("CoolTerm rows parsed (date 100826):", len(ct_epoch))
print("time range:", ct_epoch.min(), ct_epoch.max())

# restrict to window covering video span with margin
mission_start_epoch = 1786347268.8
video_start_epoch = mission_start_epoch + 261.4
video_end_epoch = video_start_epoch + 182.2667
mask = (ct_epoch >= mission_start_epoch - 5) & (ct_epoch <= video_end_epoch + 5)
print("rows in video window (+/-15s margin):", mask.sum())

sel = dict(
    epoch=ct_epoch[mask],
    do=np.array(ct_do, dtype=object)[mask],
    ph=np.array(ct_ph, dtype=object)[mask],
    temp=np.array(ct_temp, dtype=object)[mask],
    cond=np.array(ct_cond, dtype=object)[mask],
    orp=np.array(ct_orp, dtype=object)[mask],
    turb=np.array(ct_turb, dtype=object)[mask],
)

for k in ['do','ph','temp','cond','orp','turb']:
    vals = [v for v in sel[k] if v is not None]
    if vals:
        print(k, "n=",len(vals), "min/mean/max=", min(vals), sum(vals)/len(vals), max(vals))
    else:
        print(k, "NO VALID VALUES")

with open('./parsed.pkl','wb') as f:
    pickle.dump(dict(
        csv_epoch=epochs, csv_x=xs, csv_y=ys, csv_lat=lats, csv_lon=lons, csv_heading=headings,
        ct=sel,
        video_start_epoch=video_start_epoch,
        mission_start_epoch=1786347268.8,
    ), f)
print("saved parsed.pkl")

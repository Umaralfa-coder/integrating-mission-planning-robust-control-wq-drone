import re, numpy as np, pandas as pd
F="/mnt/user-data/uploads/CoolTerm_Capture__Untitled_0__2026-08-12_10-42-38-203.txt"
pat=re.compile(r'^060826;(\d+\.\d+);([\-0-9.]+);([\-0-9.]+);([\-0-9.]+);([^;]*);([^;]*);([^;]*);([^;]*);?')
tp=re.compile(r'\$PVHY2,\d+,\d+,([\-0-9.]+),')
cat=re.compile(r'^(-?\d+\.\d+)(-\d+\.\d+)$')
def tod(v):
    v=float(v); return int(v//10000)*3600+int((v%10000)//100)*60+v%100
def dd(v):
    v=float(v)
    if v==0: return np.nan
    d=int(v/100); return d+(v-d*100)/60
rows=[]
for line in open(F,encoding='utf-8',errors='replace'):
    m=pat.match(line)
    if not m: continue
    t,lat,lon,alt,rs,s1,s2,s3=m.groups()
    r=dict(t=tod(t),lat=dd(lat),lon=dd(lon),Temp=np.nan,DO=np.nan,pH=np.nan,ORP=np.nan,Cond=np.nan,Turb=np.nan)
    try:
        f=s1.split(',');   r['Temp']=float(f[1]); r['DO']=float(f[3])
    except Exception: pass
    try:
        f=s2.split(','); r['pH']=float(f[2]); mm=cat.match(f[3]); r['ORP']=float(mm.group(1)) if mm else np.nan
    except Exception: pass
    try:
        f=s3.split(','); r['Cond']=float(f[2])
    except Exception: pass
    mt=tp.search(rs)
    if mt: r['Turb']=float(mt.group(1))
    rows.append(r)
d=pd.DataFrame(rows); d.to_pickle("aug6_raw.pkl")
print(d.shape); print(d.describe().T.to_string())
print("gps valid:",d.lat.notna().sum())

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.font_manager import FontProperties
d=pd.read_pickle("aug6_raw.pkl").sort_values('t').reset_index(drop=True)
d['h']=d.t/3600; d=d[d.h<15.82].copy(); d['m']=(d.t//60).astype(int)   # drop the recovery at the end (out of water)
b=d.groupby('m').median(numeric_only=True); b['h']=b.t/3600
b['anom']=b.Turb>50
# merge anomalous minutes into episodes (gap <= 3 min)
am=b.index[b.anom].values; ep=[]
s=pr=am[0]
for x in am[1:]:
    if x-pr>3: ep.append((s,pr)); s=x
    pr=x
ep.append((s,pr)); ep=[(a*60/3600,(c+1)*60/3600) for a,c in ep if c-a>=1 or True]
print("episodes (h):",[(round(a,2),round(c,2)) for a,c in ep])
F="/usr/share/fonts/truetype/dejavu/"
TF=FontProperties(fname=F+"DejaVuSansCondensed-Bold.ttf",size=12); LB=FontProperties(fname=F+"DejaVuSansCondensed.ttf",size=11)
panels=[("Temp","Temperature (°C)","#1f77b4",None),("DO","Dissolved O$_2$ (mg/L)","#2a9d8f",None),
        ("pH","pH","#7b5ea7",None),("Turb","Turbidity (NTU, log)","#c2410c","log"),
        ("Cond","Conductivity (mS/cm)","#5f6b7a",None),("ORP","ORP (mV)","#8a6d1d",None)]
fig,axs=plt.subplots(6,1,figsize=(10,11.2),sharex=True,dpi=200,gridspec_kw=dict(hspace=0.14))
for i,(ax,(k,lab,col,sc)) in enumerate(zip(axs,panels)):
    for a,c in ep: ax.axvspan(a,c,color="#d62728",alpha=.13,lw=0,zorder=0)
    ax.plot(d.h,d[k],color=col,alpha=.28,lw=.5,zorder=2,rasterized=True)
    ax.plot(b.h,b[k],color=col,lw=1.4,zorder=3)
    ax.set_ylabel(lab,fontproperties=LB); ax.tick_params(labelsize=10)
    ax.grid(axis="y",lw=.4,alpha=.4); ax.set_xlim(8.0,15.85)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.text(-0.085,1.0,f"({'abcdef'[i]})",transform=ax.transAxes,fontsize=14,fontweight="bold",va="top",ha="right")
    if k in ("Cond","ORP"):
        ax.axvspan(8.0,8.75,color="0.5",alpha=.12,lw=0,zorder=0)
        ax.text(8.52 if k=="ORP" else 8.38,.18 if k=="ORP" else .06,"probe\nwarm-up",transform=ax.get_xaxis_transform(),fontsize=9,ha="center",va="bottom",color="#444")
    if sc=="log":
        ax.set_yscale("log"); ax.set_ylim(2,8000)
        ax.axhline(50,color="k",lw=.8,ls="--",zorder=4); ax.text(8.03,58,"anomaly threshold: 50 NTU (1-min median)",fontsize=9,va="bottom")
        ax.axhline(10,color="#555",lw=.8,ls=":",zorder=4); ax.text(8.03,11.5,"typical baseline ≈ 10 NTU",fontsize=9,va="bottom",color="#444")
        pk=d.loc[d.Turb.idxmax()]; ax.annotate(f"max {pk.Turb:,.0f} NTU",xy=(pk.h,pk.Turb),xytext=(pk.h-2.2,2200),fontsize=10,fontweight="bold",
            arrowprops=dict(arrowstyle="->",lw=1),va="center")
axs[-1].set_xlabel("Time of day, Aug 6, 2026 (logged clock, hh:mm)",fontproperties=LB)
axs[-1].xaxis.set_major_locator(MultipleLocator(1)); axs[-1].xaxis.set_minor_locator(MultipleLocator(.25))
axs[-1].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{int(x):02d}:{int(round((x%1)*60)):02d}"))
fig.suptitle("Water quality over time, Aug 6, 2026 mission (red bands: turbidity anomaly)",fontproperties=TF,y=0.905)
fig.savefig("aug6_timeseries.png",bbox_inches="tight",pad_inches=.06,facecolor="white",dpi=250)
fig.savefig("aug6_timeseries.pdf",bbox_inches="tight",pad_inches=.06,facecolor="white")

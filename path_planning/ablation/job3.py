import sys, time, json, os
area, idx, fac = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aoi4" if area=="aoi4" else "."))
sys.path.insert(0,'.')
exec(open('optimize_script_v2.py').read().split("if __name__")[0])
W=(0.35,0.25,0.25,0.15); w=list(W); S=float(sys.argv[4])

s=sum(w); w=tuple(x/s for x in w)
t0=time.time()
opt=FastGeneticAIOptimizer(area+'.txt', population_size=20, max_generations=40, quiet_mode=True, weights=w)
opt.bounds=[(0,180),(S,S+0.001),(1,1.001),(-20,20),(-20,20)]+[(-20,20)]*3
bp,bf,h=opt.optimize()
r=opt.get_path_metrics(bp['theta'],bp['width'],1)
m=r['metrics']
out=dict(area=area,idx=idx,fac=fac,weights=w,theta=float(bp['theta']),width=float(bp['width']),
 dist=float(m['total_distance']),cov=float(m['coverage']),fitness=float(bf),smooth=float(m['smoothness']),info=float(m['info_gain']),time=time.time()-t0)
json.dump(out,open(f"hd_{area}_{S}.json","w")); print(out)

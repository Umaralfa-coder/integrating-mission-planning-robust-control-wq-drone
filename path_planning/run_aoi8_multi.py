import sys, time, pickle
sys.path.insert(0,'.')
exec(open('optimize_script_v2.py').read().split("if __name__")[0])
t0=time.time()
opt = FastGeneticAIOptimizer('aoi8.txt', population_size=20, max_generations=40, quiet_mode=False)
b=list(opt.bounds); b[0]=(0,180); b[1]=(0.5,4); b[2]=(1,8.49); opt.bounds=b
print("bounds",opt.bounds, flush=True)
best_params, best_fitness, hist = opt.optimize()
r = opt.get_path_metrics(best_params['theta'], best_params['width'], best_params['num_clusters'])
print("AOI8 MULTI RESULT:", best_params, r['metrics']); print("TIME:", time.time()-t0)
pickle.dump({'params':best_params,'metrics':r['metrics']}, open('aoi8_multi_result.pkl','wb'))

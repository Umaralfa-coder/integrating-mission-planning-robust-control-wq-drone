import sys, time, pickle
sys.path.insert(0,'.')
exec(open('optimize_script_v2.py').read().split("if __name__")[0])
t0=time.time()
opt = FastGeneticAIOptimizer('aoi8.txt', population_size=20, max_generations=40, quiet_mode=False)
opt.bounds = [(0, 180), (0.5, 4), (1, 1.001), (-20, 20), (-20, 20)] + [(-20, 20)]*3
best_params, best_fitness, hist = opt.optimize()
r = opt.get_path_metrics(best_params['theta'], best_params['width'], 1)
print("AOI8 RESULT:", best_params, r['metrics']); print("TIME:", time.time()-t0)
pickle.dump({'params':best_params,'metrics':r['metrics']}, open('aoi8_de_result.pkl','wb'))

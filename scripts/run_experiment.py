import numpy as np
import pandas as pd
import os,re, time, sys
from datetime import datetime
import pygmo as pg
from scipy.stats import pearsonr
from pbrl_emoa.preference import value_functions as vf
from pbrl_emoa.problems import core as problems
from pbrl_emoa.preference import machine_dm as MachineDM
from pbrl_emoa.algorithms import pbrl_emoa as ddpgemoa
from pbrl_emoa.operators.ea_operators import problem_ideal_nadir



idi, test, mode, interactions, run =[int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])]
g, s, q= [sys.argv[6],sys.argv[7],int(sys.argv[8])]
factor=sys.argv[9]
pop_size=400
exa=10
tests_file='configs/experiments/journal_params.csv'
df=pd.read_csv (tests_file, index_col='test')
if s=='-':
    sigma=0
elif s=='sig_rnd':
    sigma=0.1
else:
    sigma=df.loc[test,s]
    
if g=='-':
    gamma=0
elif g=='gam_rnd':
    gamma=0.4
else:
    gamma=df.loc[test,g]        



df['ideal']=[np.around(np.asarray(eval(re.sub(' +',',',re.sub('\[ ','[',x)))),5) for x in df['ideal'].values]
df['nadir']=[np.around(np.asarray(eval(re.sub(' +',',',re.sub('\[ ','[',x)))),5) for x in df['nadir'].values]
df['lambd']=[np.around(np.asarray(eval(re.sub(' +',',',x))),5) if x != "-" else x for x in df['lambd'].values]
df['alpha']=[np.around(np.asarray(eval(re.sub(' +',',',x))),5) if x != "-" else x for x in df['alpha'].values]
df['beta']=[np.around(np.asarray(eval(re.sub(' +',',',re.sub('\[ ','[',x)))),5) if x != "-" else x for x in df['beta'].values]
df['tau']=[np.around(np.asarray(eval(re.sub(' +',',',x))),5) for x in df['tau'].values]
df['w']=[np.asarray(eval(re.sub(' +',',',x))) for x in df['w'].values]



df=df.loc[test]

tau=df["tau"]
beta=df["beta"]
alpha=df["alpha"]
lambd=df["lambd"]
case=int(df["case"])
ideal=df["ideal"]
nadir=df["nadir"]
fdim=int(df["fdim"])
weights=df["w"]
columns=['test','prob_id',"dim","fdim","case",'transformed',"optimum answer","optimum utility","Worst answer"
                 ,"Worst Utility","tau","alpha","beta","lambd","ideal","nadir","w","sig_25","sig_50","sig_75","gam_25",
                 "gam_50","gam_75","utility",'Run', 'Mode', 'NoOfInteractions','vf','f', 'x', 'algo', 'sigma', 'gamma', 'q','s', 'g', 'time']
results_df = pd.DataFrame(columns=columns)
prob_id=df['prob_id']
if prob_id==7:
    problem = pg.problem(pg.dtlz(prob_id = int(df['prob_id']), dim = int(df["dim"]), fdim = fdim, alpha = 100))

elif prob_id==1:
    pr = pg.dtlz(prob_id = int(df['prob_id']), dim = int(df["dim"]), fdim = fdim, alpha = 100)
    problem =problems.bounded_prob(pr,0.25,0.75)

elif prob_id==6:
    problem = pg.problem(pg.dtlz(prob_id = int(df['prob_id']), dim = int(df["dim"]), fdim = fdim, alpha = 100))
    
elif prob_id==2:
    pr = pg.dtlz(prob_id = int(df['prob_id']), dim = int(df["dim"]), fdim = fdim, alpha = 100)
    problem =problems.bounded_dtlz2(pr,0,1)


utility=df['utility']
scaling= True if utility in ['st','tst'] else False
if utility=='st':
    u=vf.stewart_value_function(w=weights*1, tau=tau*1, alpha=alpha*1, beta=beta*1, lambd=lambd*1, delta=np.full(fdim,0))
elif utility=='lin':
    u=vf.linear_value_function(weights*1, np.zeros(fdim))
elif utility=='tch':
    u=vf.tchebycheff_value_function(weights*1, np.zeros(fdim))
elif utility=='quad':
    u=vf.quadratic_value_function(weights*1, np.zeros(fdim))    
elif utility =='tst':
    u=vf.stewart_value_function_transformed(w=weights*1, tau=tau*1, alpha=alpha*1, beta=beta*1, lambd=lambd*1, delta=np.full(fdim,0))
else:
    print('utility function is not defined')


mdm=MachineDM.MachineDM(problem, u, mode, gamma, sigma, q, np.full(df["fdim"],0.5),scaling=scaling, ideal= ideal, nadir=nadir)
# print(weights)
##BCEMOA Run
# pop = pg.population(problem, size = pop_size, seed=run**5)
# m_m = 1 / problem.get_nx()
# algo = pg.algorithm(bcemoa.bcemoa(
# mdm,total_gen=200, gen = 80, cr = 0.99, eta_c = 10., m = m_m, eta_m = 50,
# seed=run**5,
# geni=32, interactions=interactions, sampleSize=exa, verbose=0))
# start=time.time()
# pop=algo.evolve(pop)
# end=time.time()
# result_df=df.copy()#pd.DataFrame( [mode], columns=['Mode'])
# result_df['f']=[np.asarray(pop.get_f()[0])]
# result_df['NoOfInteractions'] = interactions
# result_df['Run'] = run
# # result_df['problemID'] = problem.get_name()
# result_df['x'] = [np.asarray(pop.get_x()[0])]
# result_df['s'] = s
# result_df['g'] = g
# result_df['vf']=mdm.value(pop.get_f()[0],1)
# # result_df['test']=test
# result_df['algo']= 'DDPGEMOA'
# result_df['sigma']= sigma
# result_df['gamma']= gamma
# result_df['q']= q
# result_df['test'] = test
# # result_df['tau']= [tau]
# # result_df['beta']=[beta]
# # result_df['alpha']=[alpha]
# # result_df['lambd']=[lambd]
# # result_df['w']=[weights]
# # result_df['case']=case
# # result_df['utility']=utility
# result_df['factor']=factor
# result_df['mode']=mode
# result_df['time']=end-start
# # result_df['optimum utility']=df.loc[test,'optimum utility']
# results_df = results_df._append(result_df, ignore_index=True,sort=False)



# ###DDPGEMOA Run
pop = pg.population(problem, size = pop_size, seed=run**5)
m_m = 1 / problem.get_nx()
algo = pg.algorithm(ddpgemoa.DDPGEMOA(
mdm,total_gen=200, gen = 80, cr = 0.99, eta_c = 10., m = m_m, eta_m = 50,
seed=run**5,
geni=32, interactions=interactions, sampleSize=exa, verbose=0))
start=time.time()
pop=algo.evolve(pop)
end=time.time()
result_df=df.copy()#pd.DataFrame( [mode], columns=['Mode'])
result_df['f']=[np.asarray(pop.get_f()[0])]
result_df['NoOfInteractions'] = interactions
result_df['Run'] = run
# result_df['problemID'] = problem.get_name()
result_df['x'] = [np.asarray(pop.get_x()[0])]
result_df['s'] = s
result_df['g'] = g
result_df['vf']=mdm.value(pop.get_f()[0],1)
# result_df['test']=test
result_df['algo']= 'DDPGEMOA'
result_df['sigma']= sigma
result_df['gamma']= gamma
result_df['q']= q
result_df['test'] = test
# result_df['tau']= [tau]
# result_df['beta']=[beta]
# result_df['alpha']=[alpha]
# result_df['lambd']=[lambd]
# result_df['w']=[weights]
# result_df['case']=case
# result_df['utility']=utility
result_df['factor']=factor
result_df['mode']=mode
result_df['time']=end-start
# result_df['optimum utility']=df.loc[test,'optimum utility']
results_df = results_df._append(result_df, ignore_index=True,sort=False)


###iTDEA Run



# mdm=MachineDM.MachineDM(problem, u, mode, gamma, sigma, q, np.full(df["fdim"],0.5),scaling=scaling, ideal= ideal, nadir=nadir)
# pop = pg.population(problem, size = pop_size, seed=run**5)
# m_m = 1 / len(pop.get_x()[0])
# if df["fdim"]==2:
#     tau_0=0.1
#     tau_H=0.00001
# elif df["fdim"]==3:
#     tau_0=0.1
#     tau_H=0.005
# else:
#     tau_0=0.5
#     tau_H=0.25
# algo = itdea.itdea(mdm, cr = 1., eta_c = 20., m = 1. / problem.get_nx(), eta_m = 20.,tau_0 = tau_0, tau_H = tau_H, H=interactions, T=80000, mode = mode, seed=run**5, verbosity=0, exa=exa)
# start=time.time()
# pop, best_not_filtered, best_filtered, save_solution=algo.evolve(pop)
# end=time.time()
# best= best_not_filtered if mode==1 else best_filtered

# result_df=df.copy()#pd.DataFrame( [mode], columns=['Mode'])
# result_df['f']=[np.asarray(best.obj)]
# result_df['NoOfInteractions'] = interactions
# result_df['Run'] = run
# # result_df['problemID'] = problem.get_name()
# result_df['x'] = [np.asarray(best.x)]
# result_df['vf']=mdm.value(best.obj,1)
# # result_df['test']=test
# result_df['algo']= 'iTDEA'
# result_df['sigma']= sigma
# result_df['gamma']= gamma
# result_df['q']= q
# result_df['s'] = s
# result_df['g'] = g
# result_df['test'] = test
# # result_df['tau']= [tau]
# # result_df['beta']=[beta]
# # result_df['alpha']=[alpha]
# # result_df['lambd']=[lambd]
# # result_df['w']=[weights]
# # result_df['utility']=utility
# result_df['factor']=factor
# result_df['mode']=mode
# # result_df['case']=case
# result_df['time']=end-start
# # result_df['optimum utility']=df.loc[test,'optimum utility']

# results_df = results_df._append(result_df, ignore_index=True,sort=False)
                

# filename = "experiments-{}-".format(idi) + ".csv"
# results_df.to_csv(filename, index=False)


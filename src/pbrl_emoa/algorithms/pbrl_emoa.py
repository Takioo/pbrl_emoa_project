# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:38:10 2024

@author: Long
"""

import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time, sys, os
from sklearn import svm,  model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from copy import copy,deepcopy
from csv import writer
# MANUEL: This should not be necessary.
from pbrl_emoa.problems.core import as_dummy
from pbrl_emoa.preference.rank_svm import utility2prefs, RankSVM
from pbrl_emoa.operators.ea_operators import (
    tournament_selection,
    polynomial_mutation,
    sbx_crossover,
    select_best_N_mo,
)
from pbrl_emoa.rl.ddpg_agent import OptimizationEnv, DDPGAgent, LambdaMART
'''
*

* @param[in] cr Crossover probability.
* @param[in] eta_c Distribution index for crossover.
* @param[in] m Mutation probability.
* @param[in] eta_m Distribution index for mutation.
* @param seed seed used by the internal random number generator (default is random)
* @throws std::invalid_argument if \p cr is not \f$ \in [0,1[\f$, \p m is not \f$ \in [0,1]\f$, \p eta_c is not in
* [1,100[ or \p eta_m is not in [1,100[.
* @param gen1: number of nsga2 generations before the first interaction of bcemoa
* @param geni: number of generations between two consecutive interactions
* @param sampleSize: number of solutions presented to the DM for pairwise comparison
* @param detection:  determines if the detection of hidden objectives and consequently the update of objs are active or not                                   
*/'''    

class DDPGEMOA:
    def __init__ (self, dm, total_gen=500, gen = 100, cr = 0.95, eta_c = 10., statedim = 50, m = 0.01, eta_m = 50.,
                  seed = 564654561, geni=20, interactions=10, sampleSize=5, verbose = 0, test=True, estimator=svm.SVR()):
        np.random.seed(seed)
        self.gen=gen
        self.nsga2 = pg.algorithm(pg.nsga2(gen=gen, cr=cr, eta_c=eta_c, m=m, eta_m=eta_m, seed=seed))
        self.geni = geni
        self.interactions=interactions
        self.mdm = dm
        self.m_cr = cr
        self.m_m = m
        self.m_eta_c = eta_c
        self.m_eta_m = eta_m
        self.sampleSize = sampleSize
        self.m_log = []
        self.total_gen=total_gen
        self.verbose = verbose
        self.test=test
        self.training_x=[]
        self.statedim = 50
        self.lambdamart = LambdaMART()
    def get_name(self):
        return 'DDPGEMOA'

    def evolve(self,pop):
        pop = self.nsga2.evolve(pop)
        save_f, save_solution, pop = self.evolvei(pop)
        ff = pop.get_f()
        mpre_index = self.mdm.most_preferred(ff)
        for i in range(len(save_solution)):
            pop.set_xf(len(pop)-i-1, save_solution[i], save_f[i])
        return pop
      
    def evolvei(self, pop):
        vf_inter=[self.mdm.value(pop.get_f()[0],1)]#
        N = len(pop)
        fdim = len(pop.get_f()[0])
        save_solution = []
        save_f = []
        if N < self.sampleSize:
            print(f"Warning: Sample size={self.sampleSize} was larger than N={N}, adjusting it")
            self.sampleSize = N
        self.total_gen-=self.gen#gen iterations have been performed in nsga2 prior to evolvei       
        prob = pop.problem
        env = OptimizationEnv(self.mdm, pop, self.statedim)
        agent = DDPGAgent(state_dim=self.statedim * fdim, action_dim=fdim, gamma=0.99, tau=0.01, noise_std=0.1)
        
        self.state = env.selected_solutions.flatten() 
        
        
        for LearningIteration in range(self.interactions):
            
            print('learning iteration{}, Best sol:{}'.format(LearningIteration, pop.get_f()[0]))
            if self.mdm.mode != 1: # rank by svm;   
                candidates = pop.get_f()
                action = agent.select_action(self.state)
                
                
                self.feature_weights = 1 + action
                self.lambdamart.train(candidates, np.dot(candidates, self.feature_weights))
                
                
                sorted_candidates = sorted(candidates, key=lambda x: -self.lambdamart.predict(x.reshape(1, -1)), reverse=True)
                
                next_state, reward = env.update_state_and_reward(self.state, sorted_candidates)
                agent.update((self.state, action, reward, next_state))
                agent.update()
                self.state = next_state
                scaler=MinMaxScaler((-1,1)).fit(pop.get_f())
               
                temp_pop = deepcopy(pop)
                index=0
                self.training_x = pop.get_x()
                rank_pref = self.mdm.setRankingPreferences(candidates)
                for tr_ind in np.argsort(rank_pref):
                    pop.set_x(index, self.training_x[tr_ind])
                    index+=1
                
                
                pref_fun= lambda x_: self.lambdamart.predict(scaler.transform(x_))#predictor.predict
            else:
                pref_fun = self.mdm.value_array
              

            if LearningIteration == self.interactions-1:
                self.geni = int(self.total_gen - (self.interactions - 1) * self.geni)
            # The NSGA2 loop with preference replacing the crowding distance
            for gens in range(self.geni):
                f = pop.get_f()
                pop_pref = pref_fun(f)
                # print(pop_pref)
                _, _, _, ndr = pg.fast_non_dominated_sorting(f)
                shuffle1 = np.random.permutation(N)
                shuffle2 = np.random.permutation(N)
                x = pop.get_x()
                # We make a shallow copy to not change the original pop.
                pop_new = deepcopy(pop)
                for i in range(0, N, 4):
                    child1, child2 = self.generate_two_children(prob, shuffle1, i, x, ndr, -pop_pref)
                    # we use prob to evaluate the fitness so that its feval
                    # counter is correctly updated
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))
                    
                    child1, child2 = self.generate_two_children(prob, shuffle2, i, x, ndr, -pop_pref)
                    pop_new.push_back(child1, prob.fitness(child1))
                    pop_new.push_back(child2, prob.fitness(child2))
                
        
                # Selecting Best N individuals
                f = pop_new.get_f()
                best_idx = select_best_N_mo(f, N, pref_fun)
                                
                assert len(best_idx) == N
                x = pop_new.get_x()[best_idx]
                f = f[best_idx]
                for i in range(len(pop)):
                    pop.set_xf(i, x[i], f[i])
                save_solution.append(x[0])
                save_f.append(f[0])
                if self.verbose>1:
                    print(f"best vf: {self.mdm.value(pop.get_f()[0])}")
            if self.verbose:
                plt.clf()
                plt.scatter(pop.get_f()[:,0], pop.get_f()[:,1])
                plt.xlim(0.0, None)
                plt.ylim(0.0, None)
                plt.pause(0.000001)
            vf_inter.append(self.mdm.value(pop.get_f()[0],1))
        return save_f, save_solution, pop

    def get_preferences(self, env, agent, lambdamart, pop, training_set, rank_pref, pairwise):
        prob=pop.problem
        start = len(training_set)
        remaining = self.sampleSize
        ndf, _, _, ndr = pg.fast_non_dominated_sorting(pop.get_f())
        if len(ndf[0])>remaining:
            indexes=np.random.choice(ndf[0], size=remaining, replace=False)
        else:
            indexes=np.random.choice(range(len(pop)), size=remaining, replace=False)
        for i in indexes:
            f = pop.get_f()[i]
            training_set.append(f)
            self.training_x.append(pop.get_x()[i])
            remaining -= 1
            if remaining == 0:
                break
            
        if self.test:
            start=0
            rank_pref=np.empty(0)
            pairwise=np.empty(0)
        tmp_rank_pref = self.mdm.setRankingPreferences(training_set[start:])
        return training_set, np.append(rank_pref, tmp_rank_pref).astype(int), np.append(pairwise, utility2prefs(tmp_rank_pref)+start).reshape(-1,2).astype(int), env, agent, lambdamart

    
    def crossover(self, problem, parent1, parent2):
        return sbx_crossover(problem, parent1, parent2, self.m_cr, self.m_eta_c)
      
    def mutate(self, problem, child):
        return polynomial_mutation(problem, child, self.m_m, self.m_eta_m)
                          
    def generate_two_children(self, problem, shuffle, i, X, ndr, pop_pref):
        parent1_idx = tournament_selection(shuffle[i], shuffle[i + 1], ndr, pop_pref) 
        parent2_idx = tournament_selection(shuffle[i + 2], shuffle[i + 3], ndr, pop_pref)
        parent1 = X[parent1_idx]
        parent2 = X[parent2_idx]
        child1, child2 = self.crossover(problem, parent1, parent2)
        child1 = self.mutate(problem, child1)
        child2 = self.mutate(problem, child2)
        return child1, child2

    def get_log(self):
        return self.m_log
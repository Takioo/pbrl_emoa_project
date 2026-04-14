#class dtlz for implementation of dtlz7 as in bcemoa (dtlz8 in Deb's first paper)
import numpy as np
from scipy.stats import norm
import pygmo as pg
import copy
from pbrl_emoa.preference import value_functions as value_function
from pbrl_emoa.operators.ea_operators import problem_ideal_nadir

class dtlz8():
    def __init__(self,fdim=10,dim=20):
        #initialize number of objectives and variables resp.
        self.m=fdim
        self.n=dim
    
    def get_bounds(self):
        return (np.full(self.n,0),np.full(self.n,1))
    def get_nix(self):
        return 0
    def get_nic(self):
        return self.m
    
    def fitness(self,x):
        m=self.m
        n=self.n
        f=np.zeros(self.m)
        for j in range(m):
            s=0
            for i in range(int(np.floor((j)*n/m)+1),int(np.floor((j+1)*n/m))):
                s+=x[i]
            f[j]=(1/(np.floor(n/m)))* s
        g=np.zeros(self.m)
        mini=np.inf
        for i in range(m):
            for j in range(m):
                if i==j:
                    continue
                if f[i]-f[j]<mini:
                    mini=f[i]-f[j]
        for j in range(m-2):
            g[j]=f[m-1]+4*f[j]-1
        
        g[m-1]=2*f[m-1]+mini-1
        #Standard inequality constraints in PyGmoare defined as <=
        g=-g
        f=np.append(f,g)
        return f
       
class bounded_prob():
    def __init__(self,prob, lb, ub):
        #initialize number of objectives and variables resp.
        self.prob=pg.problem(prob)
        self.lb=lb
        self.ub=ub
    
    def get_bounds(self):
        return (np.full(self.prob.get_nx(),self.lb),np.full(self.prob.get_nx(),self.ub))
    def get_nix(self):
        return self.prob.get_nix()
    def get_nic(self):
        return self.prob.get_nic()
    def fitness(self,x):
        return self.prob.fitness(x)
    def get_nobj(self):
        return self.prob.get_nobj()
    def get_name(self):
        return  self.prob.get_name()
    def get_nx(self):
        return self.prob.get_nx()
    def get_nec(self):
        return self.prob.get_nec()
    
class bounded_dtlz2():
    def __init__(self,prob, lb=0, ub=1):
        #initialize number of objectives and variables resp.
        self.prob=pg.problem(prob)
        self.lb=lb
        self.ub=ub
    
    def get_bounds(self):
        return (np.full(self.prob.get_nx(),self.lb),np.full(self.prob.get_nx(),self.ub))
    def get_nix(self):
        return self.prob.get_nix()
    def get_nic(self):
        return self.prob.get_nic()
    def fitness(self,x):
        return self.prob.fitness(x/2+0.25)
    def get_nobj(self):
        return self.prob.get_nobj()
    def get_name(self):
        return  self.prob.get_name()
    def get_nx(self):
        return self.prob.get_nx()
    def get_nec(self):
        return self.prob.get_nec()    
    
class knapsack():
    def __init__(self,c,w,p,m=10,n=100):#m is the number of knapsacks/objectives
        self.m=m
        self.n=n
        self.p=p
        self.c=c
        self.w=w
    
    def get_bounds(self):
        return (np.full(self.n,0),np.full(self.n,self.m))
    def get_nix(self):
        return self.n

    def get_nic(self):
        return self.m
    
    def fitness(self,x):
        f=np.zeros(self.m)
        g1=np.zeros(self.m)
        for i in x:
            if x[i]==0:
                continue
            f[x[i]]+=self.p[i]
            g1[x[i]]+=self.w[i]
        g1=g1-self.c
        return np.append(f,g1)

class dummy:
    def __init__(self, prob, dv):#To be fixed:, f_lbound, f_ubound
        self.prob = prob
        # Create a copy to not modify the original vector.
        self.dv = dv*1

    def get_nobj(self):
        return len(self.dv)

    def original_fitness(self,x):
        return self.prob.fitness(x)

    def fitness(self,x):
        return self.original_fitness(x)[self.dv]

    def get_bounds(self):
        return self.prob.get_bounds()

    # def get_f_bounds(self):
    #     return self.f_bounds 

    def get_name(self):
        return "dummy " + self.prob.get_name()

    def get_nix(self):
        return self.prob.get_nix()

    def get_nx(self):
        return self.prob.get_nx()

    def get_nic(self):
        return self.prob.get_nic()

    def get_nec(self):
        return self.prob.get_nec()

    def get_dv(self):
        return self.dv

    def set_dv(self, new_dv):
        self.dv = new_dv

    def get_original_f(self, population):
        x = population.get_x()
        f = np.apply_along_axis(self.original_fitness, 1, x)
        return f

class dummy2:
    def __init__(self, prob1, prob2, dv):#The two problems are assumed to have the same dim and bounds
        self.prob1 = prob1
        self.prob2 = prob2
        # Create a copy to not modify the original vector.
        self.dv = dv*1

    def get_nobj(self):
        return len(self.dv)

    def original_fitness(self,x):
        return np.append(self.prob1.fitness(x),self.prob2.fitness(x))

    def fitness(self,x):
        return self.original_fitness(x)[self.dv]

    def get_bounds(self):
        return self.prob1.get_bounds()

    # def get_f_bounds(self):
    #     return self.f_bounds 

    def get_name(self):
        return "dummy " + self.prob1.get_name()+ " + " + self.prob2.get_name()

    def get_nix(self):
        return self.prob1.get_nix()

    def get_nx(self):
        return self.prob1.get_nx()

    def get_nic(self):
        return self.prob1.get_nic()+self.prob2.get_nic()

    def get_nec(self):
        return self.prob1.get_nec()+self.prob2.get_nec()

    def get_dv(self):
        return self.dv

    def set_dv(self, new_dv):
        self.dv = new_dv

    def get_original_f(self, population):
        x = population.get_x()
        f = np.apply_along_axis(self.original_fitness, 1, x)
        return f

class vf_prob: #finds the best solution accorrding to the vf
    def __init__(self, prob, vf):
        self.prob = prob
        self.vf=vf

    def get_nobj(self):
        return 1

    def fitness(self,x):
        return [self.vf.value(self.prob.fitness(x))]

    def get_bounds(self):
        return self.prob.get_bounds()


    def get_nix(self):
        return self.prob.get_nix()

    def get_nx(self):
        return self.prob.get_nx()

    def get_nic(self):
        return self.prob.get_nic()

    def get_nec(self):
        return self.prob.get_nec()

def as_dummy(problem):
    return problem.extract(dummy)

def get_problem(name, nobj, dim):
    if "dtlz" in name:
        prob_id = int(name.replace("dtlz", ""))
        return pg.problem(pg.dtlz(prob_id = prob_id, dim = dim, fdim = nobj, alpha = 100))
    elif "zdt" in name:
        prob_id = int(name.replace("zdt", ""))
        return pg.problem(pg.zdt(prob_id = prob_id))
    else:
        assert False
        
        
class stewart_params: #finds the best solution accorrding to the vf
    def __init__(self, prob, pf, case, point, alpha_beta_low_levels=True, transformed=False):
        self.prob = prob
        self.pf=pf*1
        self.nadir=pf.max(axis=0)
        self.ideal=pf.min(axis=0)
        _, self.f_max=problem_ideal_nadir(prob)
        self.nobj=prob.get_nobj()
        self.point=0.95*(point-self.ideal)/(self.nadir-self.ideal)
        self.transformed=transformed
        
        #scaling all pf points to [0,1] in order to use sigmoid function
        for i in range(len(self.pf)):
            # obj=self.pf[i]
            self.pf[i]=0.95*(self.pf[i]-self.ideal)/(self.nadir-self.ideal)
            # obj_[obj_>0.95]=0.95+0.05*obj[obj_>0.95]/self.f_max[obj_>0.95]
            # obj_[obj_<0]=0
            # self.pf[i]=obj_
            
        #defining bounds
        self.w_min=np.full(self.nobj,1/(self.nobj*5))
        self.w_max=np.full(self.nobj,0.95)
        if self.transformed:
            if case==1:case=3 
            if case==3:case=1 
            if case==4:case=2 
            if case==2:case=4 
        if case==1:
            self.tau_min=np.full(self.nobj,0.1)
            self.tau_max=np.full(self.nobj,0.4)
            self.lambd_min=np.full(self.nobj,0.1)
            self.lambd_max=np.full(self.nobj,0.4)
        elif case==2:
            self.tau_min=np.full(self.nobj,0.1)
            self.tau_max=np.full(self.nobj,0.4)
            self.lambd_min=np.full(self.nobj,0.6)
            self.lambd_max=np.full(self.nobj,0.9)
        elif case==3:
            self.tau_min=np.full(self.nobj,0.6)
            self.tau_max=np.full(self.nobj,0.9)
            self.lambd_min=np.full(self.nobj,0.1)
            self.lambd_max=np.full(self.nobj,0.4)
        elif case==4:
            self.tau_min=np.full(self.nobj,0.6)
            self.tau_max=np.full(self.nobj,0.9)
            self.lambd_min=np.full(self.nobj,0.6)
            self.lambd_max=np.full(self.nobj,0.9)
        else:
            self.tau_min=np.full(self.nobj,0.1)
            self.tau_max=np.full(self.nobj,0.9)
            self.lambd_min=np.full(self.nobj,0.1)
            self.lambd_max=np.full(self.nobj,0.9)
            
        if alpha_beta_low_levels==True:
            self.alpha_min=np.full(self.nobj,10)
            self.alpha_max=np.full(self.nobj,16)
            self.beta_min=np.full(self.nobj,1)
            self.beta_max=np.full(self.nobj,7)
        elif alpha_beta_low_levels==False:
            self.alpha_min=np.full(self.nobj,16)
            self.alpha_max=np.full(self.nobj,22)
            self.beta_min=np.full(self.nobj,8)
            self.beta_max=np.full(self.nobj,14)
        else:
            self.alpha_min=np.full(self.nobj,10)
            self.alpha_max=np.full(self.nobj,22)
            self.beta_min=np.full(self.nobj,1)
            self.beta_max=np.full(self.nobj,14)
        
    def get_nobj(self):
        return 1
    
    def get_name(self):
        return 'Stewart_params'

    def fitness(self,x):
        w=x[:self.nobj]/sum(x[:self.nobj])
        tau=x[self.nobj:2*self.nobj]
        lambd=x[2*self.nobj:3*self.nobj]
        alpha=x[3*self.nobj:4*self.nobj]
        beta=x[4*self.nobj:]
        if self.transformed:
            uf=value_function.stewart_value_function_transformed(w, tau, alpha, beta, lambd)
        else:
            uf=value_function.stewart_value_function(w, tau, alpha, beta, lambd)
        pf_u = np.apply_along_axis(uf.value, axis=1, arr = self.pf)
        max_u=pf_u.max()
        min_u=pf_u.min()
        best=self.pf[pf_u.argmin()]
        penalty=0 if max_u-min_u>=.1 else 1./ (max_u-min_u)
        return [sum(abs(self.point-best))+penalty]
       
        # ce1=10*sum(obj[obj>0.95])
        # ce2=1000*sum(0.05-obj[obj<0.05])
        # return [min_u-max_u+ce1+ce2]

    def get_bounds(self):
        # return np.asarray([np.hstack([self.tau_min, self.lambd_min, self.alpha_min, self.beta_min]), np.hstack([self.tau_max, self.lambd_max, self.alpha_max, self.beta_max])])
        return (list(np.hstack([self.w_min, self.tau_min, self.lambd_min, self.alpha_min, self.beta_min])), list(np.hstack([self.w_max, self.tau_max, self.lambd_max, self.alpha_max, self.beta_max])))

    def get_nix(self):
        return 2*self.nobj
    

class MNK:
    def __init__(self, M,N,K):# prob
        assert K<N, "K should be strictly less than N"
        # Create a copy to not modify the original vector.
        self.M = M
        self.N = N
        self.K = K
        
        #define epistatic structure(ES): M*N*k structure
        self.links=np.empty([self.M, self.N, self.K+1],dtype=int)
        for m in range (self.M):
            for n in range (self.N):
                self.links[m,n]=np.append(np.random.choice([x for x in np.arange(self.N) if x != n], self.K, replace=False),n)
                
        #define the contributions of each pattern to different objectives
        #we define a uniformly distributed numbers over a vector with size M,N,(2**(K+1)) 
        #the pattern in the chromosome is binary, we turn it to integer to access the contribution of the
        #pattern like cont[m,n,int(pattern)]
        
        self.tables=np.random.uniform(size=[M,N,2**(K+1)])

    def get_nobj(self):
        return self.M

    def fitness(self,x):
        fit=[0]*self.M
        for m in range (self.M):
            for n in range (self.N):
                pattern=self.links[m][n]
                sol=x[pattern].astype(int)
                fit[m]+=(self.tables[m,n,int('0b'+''.join([str(el) for el in sol]),2)])/self.N
            
                
                
        return fit

    def get_bounds(self):
        return ([0]*self.N, [1]*self.N)


    def get_name(self):
        return "MNK problem "

    def get_nix(self):
        return self.N

    # def get_nx(self):
    #     return self.N

    # def get_nic(self):
    #     return 0

    # def get_nec(self):
    #     return 0


class rMNK:
    def __init__(self,r, M,N,K):
        assert K<N, "K should be strictly less than N"
        assert r>-1/(M-1), "rho must be greater than -1/(M-1), where M is the number of objectives "
        self.R=np.full([M,M],r)
        for i in range(M):
            self.R[i,i]=1 
        self.RR=2 * np.sin(self.R*np.pi/6  )
        self.M = M
        self.N = N
        self.K = K
        
        #define epistatic structure(ES): M*N*k structure
        self.links=np.empty([self.M, self.N, self.K+1],dtype=int)
        for m in range (self.M):
            for n in range (self.N):
                self.links[m,n]=np.append(np.random.choice([x for x in np.arange(self.N) if x != n], self.K, replace=False),n)
                
        #define the contributions of each pattern to different objectives
        #we define a normally distributed numbers over a vector with size N*(2^(K+1)), M
        #the pattern in the chromosome is binary, we turn it to integer to access the contribution of the
        #pattern like cont[m,n,int(pattern)]
        self.tables=norm.cdf(np.random.multivariate_normal(np.zeros(self.M), self.RR, N*2**(K+1)))
        # =np.random.uniform(size=[M,N,2**(K+1))

    def get_nobj(self):
        return self.M

    def fitness(self,x):
        fit=[0]*self.M
        for m in range (self.M):
            for n in range (self.N):
                pattern=self.links[m][n]
                sol=x[pattern].astype(int)
                fit[m]+=self.tables[n*2**(self.K+1)+int('0b'+''.join([str(el) for el in sol]),2),m]/self.N
            
                
                
        return fit

    def get_bounds(self):
        return ([0]*self.N, [1]*self.N)


    def get_name(self):
        return "rMNK problem "

    def get_nix(self):
        return self.N     
    
    
    
    
class uv_params_hidden: #finds the best solution accorrding to the vf
    def __init__(self, dummy_prob, point):
        self.prob = dummy_prob
        self.nobj=dummy_prob.get_nobj()
        self.point=point
        
        

        # The initial population
        pop = pg.population(dummy_prob, size = 200)
        algo=pg.algorithm(pg.nsga2(gen=500))
        pop=algo.evolve(pop)
        f=pop.get_f()
        ndf, _, _, ndr = pg.fast_non_dominated_sorting(f)
        self.pf=f[ndf[0]]
        # self.u=u
        
        #defining bounds
        self.w_min=np.full(self.nobj,0.01)
        self.w_max=np.full(self.nobj,0.99)
        self.ideal=np.full(self.nobj,0.0)
        # if dummy_prob.prob.get_name()[-1]=='1' or dummy_prob.prob.get_name()[-1]=='2':
        #     self.nadir=np.full(self.nobj,0.5)
        # else:
        
        #     self.nadir=self.pf.max(axis=0)
    def get_nobj(self):
        return 1
    
    def get_name(self):
        return 'uv_params_hidden'

    def fitness(self,x):
        w=x/sum(x)#[:self.nobj]/sum(x[:self.nobj])
        # ideal=x[self.nobj:2*self.nobj]
        
        min_u=100000
        max_u=0
        u=value_function.quadratic_value_function(w, self.ideal)
        # u=value_function.tchebycheff_value_function(w, self.ideal)

        for obj in self.pf:
            total=u.value(obj)
    
            if total>max_u:
                max_u=total
            if total<min_u:
                min_u=total
                best=obj
        print(best)
        return [-min(best)]#[sum(abs(self.point-best))]#

    def get_bounds(self):

        return (list(np.hstack([self.w_min])), list(np.hstack([self.w_max])))

    # def get_nix(self):
    #     return 2*self.nobj
    



class uv_params: #finds the best solution accorrding to the vf
    def __init__(self, point, pf, uf):
        self.nobj=len(pf[0])
        self.point=point
        self.pf=pf
        self.uf=uf
        
        #defining bounds
        self.w_min=np.full(self.nobj,0.01)
        self.w_max=np.full(self.nobj,0.99)
        self.ideal=np.full(self.nobj,0.0)

    def get_nobj(self):
        return 1
    
    def get_name(self):
        return 'uv_params_hidden'

    def fitness(self,x):
        w=x/sum(x)
        u=self.uf(w, self.ideal)
        pf_u = np.apply_along_axis(u.value, axis=1, arr = self.pf)
        max_u=pf_u.max()
        min_u=pf_u.min()
        best=self.pf[pf_u.argmin()]
        penalty=0 if max_u-min_u>=.1 else 1./ (max_u-min_u)
        return [sum(abs(self.point-best))+penalty] #[-min(best)]#[sum(abs(self.point-best))]#

    def get_bounds(self):
        return (list(np.hstack([self.w_min])), list(np.hstack([self.w_max])))





## For testing
# problem = pg.dtlz(prob_id = 1, dim = 5, fdim = 4, alpha = 100)
# dummyVector=np.asarray([0,1,0,1])
# prob=pg.problem(dummy(problem, dummyVector))
# x = np.array([0.5,0.5,0.5,0.5,0.5])
# print(prob.fitness(x))
# print(prob.extract(dummy).original_fitness(x))
# prob.extract(dummy).dv = [1, 0, 1, 0]
# print(prob.fitness(x))
# print(prob.extract(dummy).original_fitness(x))
# print(prob.get_fevals())



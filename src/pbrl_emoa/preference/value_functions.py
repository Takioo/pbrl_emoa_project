############ Utility / value functions
import numpy as np

class value_function:
    def __init__(self, weights):
        # Should we check only or enforce it with a warning ourselves?
        # We store copies to avoid being affected by the caller
        assert np.isclose(sum(weights),1.)
        self.weights = np.array(weights)*1
        
        
class linear_value_function(value_function):

    def value(self, obj):
        return np.inner(self.weights, obj)

    def get_name(self): return "linear"
    def get_full_name(self): return f"linear(w={self.weights})"

class stewart_value_function(value_function):
    def __init__(self, w, tau, alpha, beta, lambd, delta=0):
        # FIXME: Add checking of input parameters
        super().__init__(w)
        self.tau = np.array(tau)*1
        self.alpha = np.array(alpha)*1
        self.beta = np.array(beta)*1
        self.lambd = np.array(lambd)*1
        self.delta = np.array(delta)*1
        
        # self.tau = 1-np.array(tau)*1
        # self.alpha = np.array(beta)*1 
        # self.beta = np.array(alpha)*1
        # self.lambd = 1-np.array(lambd)*1
        # self.delta = np.array(delta)*1


    def value(self, obj):#_
        # obj=1-obj_
        """#/ (c) a shift in the reference levels tau_i from the ‘ideal’
           positions by an amount \delta, which may be
           positive or negative (and which is also a
           specified model parameter)."""
        assert np.all(obj >= 0)
        assert np.all(obj <= 1.0001)
        tau_mod = self.tau + self.delta
        subtotal = np.array(self.lambd)*1
        losses = obj <= tau_mod
        
        for i in range(len(obj)):
            if (losses[i]):
                subtotal[i] *= (np.exp(self.alpha[i] * obj[i]) - 1.0) / (np.exp(self.alpha[i] * tau_mod[i]) - 1.0)
            else:
                subtotal[i] += (1.0 - self.lambd[i]) * (1.0 - np.exp(-self.beta[i] * (obj[i] - tau_mod[i]))) / (1.0 - np.exp(-self.beta[i] * (1.0 - tau_mod[i])))
                
        total = np.inner(self.weights, subtotal)
        # total=1-total
        return total

    def get_name(self): return "stewart"
    def get_full_name(self):
        return f"stewart(w={self.weights}, tau={self.tau}, alpha={self.alpha}, beta={self.beta}, lambd={self.lambd}, delta={self.delta})"
    

class stewart_value_function_transformed(value_function):
    def __init__(self, w, tau, alpha, beta, lambd, delta=0):
        # FIXME: Add checking of input parameters
        super().__init__(w)
        self.tau = np.array(tau)*1
        self.alpha = np.array(alpha)*1
        self.beta = np.array(beta)*1
        self.lambd = np.array(lambd)*1
        self.delta = np.array(delta)*1

    def value(self, obj):#_
        # obj=1-obj_
        """#/ (c) a shift in the reference levels tau_i from the ‘ideal’
           positions by an amount \delta, which may be
           positive or negative (and which is also a
           specified model parameter)."""
        assert np.all(obj >= 0)
        assert np.all(obj <= 1.0001)
        tau_mod = self.tau + self.delta
        
        uv=0
            
        for i in range(len(obj)):
            if (obj[i]<tau_mod[i]):
                uv+=self.weights[i]*( self.lambd[i]+(1.0 - self.lambd[i]) * (1.0 - np.exp(-self.beta[i] * (tau_mod[i]-obj[i]))) / (1.0 - np.exp(-self.beta[i] * tau_mod[i])))
            else:
                uv+=self.weights[i]*(self.lambd[i]* (np.exp(self.alpha[i] * (1-obj[i])) - 1.0) / (np.exp(self.alpha[i] * (1-tau_mod[i])) - 1.0))
        
        return 1-uv

    def get_name(self): return "stewart_transformed"
    def get_full_name(self):
        return f"stewart_transformed(w={self.weights}, tau={self.tau}, alpha={self.alpha}, beta={self.beta}, lambd={self.lambd}, delta={self.delta})"
   

    
class quadratic_value_function(value_function):
    def __init__(self, weights, ideal):
        super().__init__(weights)
        self.ideal = np.array(ideal)*1
        assert len(self.ideal) == len(self.weights)
        
    def value(self, obj):
        assert len(self.weights) == len(obj)
        return np.sum(np.power(self.weights * (obj - self.ideal), 2))
    
    def get_name(self): return "quadratic"
    def get_full_name(self):
        return f"quadratic(w={self.weights}, ideal={self.ideal})"
    
class tchebycheff_value_function(value_function):
    def __init__(self, weights, ideal):
        super().__init__(weights)
        self.ideal = np.array(ideal)*1
        assert len(self.ideal) == len(self.weights)
        
    def value(self, obj):
        return np.max(self.weights * np.abs(obj - self.ideal))

    def get_name(self): return "tchebycheff"
    def get_full_name(self):
        return f"tchebycheff(w={self.weights}, ideal={self.ideal})"

class polynomial(value_function):
    def __init__(self, y):
        """ y: a string containing a python mathematical expression using 'x' """
        self.y = y
        
    def value(self, x):
        # MANUEL: What is this doing?
        # if  (type(x) is np.ndarray) and (type(x[0]) is np.ndarray):
        #     x=x[0]
        ans = eval(self.y)
        return ans

    def get_name(self): return "polynomial"
    def get_full_name(self):
        return f"polynomial({self.y})"

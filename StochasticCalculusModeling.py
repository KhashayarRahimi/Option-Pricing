"""
Here we try to model the option prices by methods in stochastic calculus as the following:

1. Black-Scholes
2. Geometric Brownian Motion
3. Black-Scholes-Merton
4. Black-Scholes-Heston
5. Binomial Option Pricing

"""
import math
import numpy as np
from scipy.stats import norm

class BlackScholes:

    def __init__(self, S_0, K, r, T, HistoricalPrice):

        self.S_0 = S_0      #current price of the basic asset
        self.K = K          #Strike Price 
        self.r = r          #Interst rate
        self.T = T/255      #Time until option Price normalized to year as th unit
        self.HistoricalPrice = HistoricalPrice  #for calculating annualized volatility of the asset's returns
    
    def Volatility(self):

        vals = self.HistoricalPrice[-255:].values

        #Calculate the log returns for volatility
        LogReturns = [np.log(vals[i+1]) - np.log(vals[i]) for i in range(1,len(vals)-1)]

        StandardDeviation = np.std(LogReturns)

        #calculate annualized volatility
        vol = StandardDeviation * np.sqrt(255)

        return vol
    
    def D_one(self):

        vol = self.Volatility()

        Variance = (vol / np.sqrt(self.T))**2

        numerator = np.log(self.S_0/self.K) + (self.r + Variance /2) * self.T

        denominator = vol

        return numerator/denominator
    
    def D_two(self):

        return self.D_one() - self.Volatility()
    
    def CallOptionPrice(self):

        d1 = self.D_one()
        
        d2 = self.D_two()

        return self.S_0 * norm.cdf(d1) - norm.cdf(d2) * self.K * np.exp(-self.r * self.T)
    
    def PutOptionPrice(self):

        d1 = self.D_one()
        d2 = self.D_two()

        return norm.cdf(-d2) * self.K * np.exp(-self.r * self.T) - self.S_0 * norm.cdf(-d1)

#----------------------------------------------------------
    """
    Example:
    ---------------------------------------------------------
    import numpy as np
    import pandas as pd
    from StochasticCalculusModeling import BlackScholes
    
    
    s_0 = GoldCoinPrice['Close']['1398/01/14']
    k = 44000000
    r = 0.5
    T = 32
    HistoricalPrice = GoldCoinPrice['Close']
    BS = BlackScholes(s_0,k,r,T,HistoricalPrice)
    
    print('====Call====')
    CallPrice = BS.CallOptionPrice()    
    print(CallPrice)

    print('====Put====')
    7692708.033953875

    PutPrice = BS.PutOptionPrice()
    print(PutPrice)
    2456753.1139931306
    """

#========================================================================


    
    #Geomteric Brownian Motion For Forecasting the Price

class GBM:

    def __init__(self, S_0, mu, sigma, N, dt=1/255):

        self.S_0 = S_0      # current price
        self.mu = mu        # mean of the price
        self.sigma = sigma  # variance of the price
        self.N = N         # duration to forecast - we want to estimate price at time 't'
        self.dt = dt        #the change step which the default is daily

    
    def S_t(self):

        # last term is a brownian motion which is a normal process with mean=0 and variance=dt
        #this term explain the stochastic part of the pricing

        Z = np.random.standard_normal(self.N)

        # Calculate the price path
        S = np.zeros(self.N)
        S[0] = self.S_0
        for t in range(1, self.N):
            S[t] = S[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * Z[t])

        return S
    
    def Expected_St(self):

        return self.S_0 * np.exp(self.mu * self.dt)
    
    def Variance_St(self):

        return (self.S_0 ** 2) * (np.exp(2 * self.mu * self.dt)) * (np.exp((self.sigma**2) * self.dt) - 1)


#----------------------------------------------------------
    """
    Example:
    ---------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from StochasticCalculusModeling import GBM
    
    
    
    s_0 = GoldCoinPrice['Close']['1398/01/14']
    N = 28
    HistoricalPrice = GoldCoinPrice['Close']['1397/01/14':'1398/01/14']
    vals = HistoricalPrice.values

    #Calculate the log returns for volatility
    LogReturns = [np.log(vals[i+1]) - np.log(vals[i]) for i in range(1,len(vals)-1)]

    StandardDeviation = np.std(LogReturns)
    mu = np.mean(LogReturns)
    #calculate annualized volatility
    vol = StandardDeviation * np.sqrt(255)
    gbm = GBM(s_0, mu, vol, N)
    s = gbm.S_t()

    # Example data
    x = GoldCoinPrice['1398/01/14':'1398/02/16'].index.tolist()
    y1 = GoldCoinPrice['Close']['1398/01/14':'1398/02/16']
    y2 = s

    # Plotting
    plt.plot(x, y1, label='actual')
    plt.plot(x, y2, label='gbm')

    # Adding labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('GBM Forecasting')
    plt.legend()

    # Display plot
    plt.show()
    """

#========================================================================
    
    #Merton Jump Diffusion Model

class MertonJumpDiffusion:

    def __init__(self,S_0, K, r, T, HistoricalPrice, m, v, lam):

        self.S_0 = S_0      # current price of the basic asset
        self.K = K          # Strike Price 
        self.r = r          # Interst rate
        self.T = T/255      # Time until option Price normalized to year as th unit
        self.HistoricalPrice = HistoricalPrice  # for calculating annualized volatility of the asset's returns
        self.m = m          # Mean of Jump Size
        self.v = v          # Standard Deviation of Jump Size
        self.lam = lam      # Number of jumps per year (intensity)
    
    def Volatility(self):

        vals = self.HistoricalPrice[-255:].values

        #Calculate the log returns for volatility
        LogReturns = [np.log(vals[i+1]) - np.log(vals[i]) for i in range(1,len(vals)-1)]

        StandardDeviation = np.std(LogReturns)

        #calculate annualized volatility
        vol = StandardDeviation * np.sqrt(255)

        return vol
    
    def D_one(self, vol, R):

        #vol = self.Volatility()

        Variance = (vol / np.sqrt(self.T))**2

        numerator = np.log(self.S_0/self.K) + (R + Variance /2) * self.T

        denominator = vol

        return numerator/denominator
    
    def D_two(self, vol, R):

        return self.D_one(vol, R) - vol
    
    def CallOptionPrice(self, vol, R):

        d1 = self.D_one(vol, R)
        
        d2 = self.D_two(vol, R)

        return self.S_0 * norm.cdf(d1) - norm.cdf(d2) * self.K * np.exp(-self.r * self.T)
    
    def PutOptionPrice(self, vol, R):

        d1 = self.D_one(vol, R)
        d2 = self.D_two(vol, R)

        return norm.cdf(-d2) * self.K * np.exp(-self.r * self.T) - self.S_0 * norm.cdf(-d1)
    

    def merton_jump_call(self):
        p = 0
        for k in range(40):
            r_k = self.r - self.lam*(self.m-1) + (k*np.log(self.m) ) / self.T
            #print('r_k', r_k)
            sigma_k = np.sqrt( self.Volatility()**2 + (k* self.v** 2) / self.T)
            #print('sigma_k',sigma_k)
            k_fact = np.math.factorial(k)
            #print('k_fact',k_fact)
            p += (np.exp(-self.m * self.lam * self.T) *\
                  (self.m * self.lam * self.T)** k / (k_fact))  * self.CallOptionPrice(sigma_k, r_k)
            #print('p',p)
        
        return p 


    def merton_jump_put(self):
        p = 0 # price of option
        for k in range(40):
            r_k = self.r - self.lam*(self.m-1) + (k*np.log(self.m) ) / self.T
            sigma_k = np.sqrt(self.Volatility()**2 + (k* self.v** 2) / self.T)
            k_fact = np.math.factorial(k) # 
            p += (np.exp(-self.m*self.lam*self.T) * (self.m*self.lam*self.T)**k / (k_fact)) \
                        * self.PutOptionPrice(sigma_k, r_k)
        return p
    
#----------------------------------------------------------
    """
    Example:
    ---------------------------------------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from StochasticCalculusModeling import MertonJumpDiffusion
    
    
    s_0 = GoldCoinPrice['Close']['1398/01/14']
    k = 44000000
    r = 0.24
    T = 32
    HistoricalPrice = GoldCoinPrice['Close']['1397/01/14':'1398/01/14']
    vals = HistoricalPrice.values

    #Calculate the log returns for volatility
    LogReturns = [np.log(vals[i+1]) - np.log(vals[i]) for i in range(1,len(vals)-1)]

    StandardDeviation = np.std(LogReturns)
    mu = np.mean(LogReturns)
    #calculate annualized volatility
    vol = StandardDeviation * np.sqrt(255)

    m = 0.0001 # meean of jump size
    v = 0.5 # standard deviation of jump
    lam = 1/2 # intensity of jump i.e. number of jumps per annum

    MJD = MertonJumpDiffusion(s_0, k,r,T, HistoricalPrice,  m, v, lam)

    MertonCall = MJD.merton_jump_call()
    print(MertonCall) ---> 11685896.628936155

    MertonPut = MJD.merton_jump_put()
    print(MertonPut) ---> 7820476.917796012
    """

#========================================================================
    
    # Heston Model Which considers the volatility itself as a stochastic process (Brownian Motion)

class HestonModel:

    def __init__(self):
        pass

    def generate_heston_paths(self, S, T, r, kappa, theta, v_0, rho, xi, 
                            steps, Npaths, return_vol=False):
        dt = T/steps
        size = (Npaths, steps)
        prices = np.zeros(size)
        sigs = np.zeros(size)
        S_t = S
        v_t = v_0
        for t in range(steps):
            WT = np.random.multivariate_normal(np.array([0,0]), 
                                            cov = np.array([[1,rho],
                                                            [rho,1]]), 
                                            size=Npaths) * np.sqrt(dt) 
            
            S_t = S_t*(np.exp( (r- 0.5*v_t)*dt+ np.sqrt(v_t) *WT[:,0] ) ) 
            v_t = np.abs(v_t + kappa*(theta-v_t)*dt + xi*np.sqrt(v_t)*WT[:,1])
            prices[:, t] = S_t
            sigs[:, t] = v_t
        
        if return_vol:
            return prices, sigs
        
        return prices
    
#----------------------------------------------------------
    """
    Example:
    ---------------------------------------------------------
    import numpy as np
    import pandas as pd
    from StochasticCalculusModeling import HestonModel
    
    
    HM = HestonModel()

    kappa = 4
    theta = 0.02
    v_0 =  0.02
    xi = 0.9
    r = 0.02
    S = 100
    Npaths =500
    steps = 32
    T = 1
    rho=0.9


    prices_pos = HM.generate_heston_paths(S, T, r, kappa, theta,
                                        v_0, rho, xi, steps, Npaths)[:,-1]
    """

    #========================================================================

    #Binomial Option Pricing

class Binomial:

    def __init__(self, S0, K , T, r, sigma, N):

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N #steps to the target - expiration

    def combos(self, n, i):
        return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))
    
    def BinomialPricing(self, type_):
        dt = self.T/self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = np.exp(-self.sigma * np.sqrt(dt))
        p = (  np.exp(self.r*dt) - d )  /  (  u - d )
        value = 0 
        for i in range(self.N+1):
            node_prob =self.combos(self.N, i)*p**i*(1-p)**(self.N-i)
            ST = self.S0*(u)**i*(d)**(self.N-i)
            if type_ == 'call':
                value += max(ST-self.K,0) * node_prob
            elif type_ == 'put':
                value += max(self.K-ST, 0)*node_prob
            else:
                raise ValueError("type_ must be 'call' or 'put'" )
        
        return value*np.exp(-self.r*self.T)

#----------------------------------------------------------
    """
    Example:
    ---------------------------------------------------------
    import numpy as np
    import pandas as pd
    from StochasticCalculusModeling import Binomial
    
    
    s_0 = GoldCoinPrice['Close']['1398/01/14']
    k = 44000000
    r = 0.024
    T = 32/255
    HistoricalPrice = GoldCoinPrice['Close']['1397/01/14':'1398/01/14']
    vals = HistoricalPrice.values

    #Calculate the log returns for volatility
    LogReturns = [np.log(vals[i+1]) - np.log(vals[i]) for i in range(1,len(vals)-1)]

    StandardDeviation = np.std(LogReturns)
    mu = np.mean(LogReturns)
    #calculate annualized volatility
    vol = StandardDeviation * np.sqrt(255)
    BP = Binomial(s_0, k, T, r, vol, N=80)
    CallPrice = BP.BinomialPricing('call')
    print('CallPrice:',CallPrice)

    PutPrice = BP.BinomialPricing('put')
    print('PutPrice:',PutPrice)

    CallPrice: 4967985.835367359
    PutPrice: 2275667.5441064755
    """
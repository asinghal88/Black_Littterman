
"""
Created on Mon Oct 25 01:24:39 2021

@author:Anshul Singhal
"""
#%%
# =============================================================================
# Loading the Libraries
# =============================================================================
import pandas as pd
import numpy as np
#%%
# =============================================================================
# #Reading the data and Calculating Covariance Matrix
# =============================================================================
df = pd.read_excel("BL -returndata.xlsx", index_col = "YearM", usecols = ['YearM','US Equity', 'Foreign EQ', 'Emerging EQ'])
df.index = pd.to_datetime(df.index, format='%Y%m', errors='coerce')
cov_matrix = df.cov()
weight_matrix = [0.5,0.4,0.1]
risk_aversion = 3
#%%
# =============================================================================
# Creating the Black Litterman Function
# =============================================================================
def Black_Litterman(tau,tau_v,p_matrix,q_matrix): 
    print("Prior Market Weights: \n",weight_matrix,"\n\n Risk Aversion Parameter: ",risk_aversion,"\n Scalar on Uncertainty in the Prior: ",tau,"\n Scalar on Manager Views: ",tau_v)
    #Prior Returns
    prior_ret = risk_aversion*np.dot(cov_matrix,weight_matrix)  
    #Uncertainty
    omega = tau_v*np.dot(p_matrix,np.dot(cov_matrix,np.array(p_matrix).T))
    #Diagonalised Values only in the matrix
    diagnolised_omega = np.diag(np.diag(omega))
    PriorViews = np.dot(p_matrix,np.dot(cov_matrix*tau,np.array(p_matrix).T))
    #Calculation of Posterior Ret
    posterior_ret = prior_ret + np.dot(np.dot(cov_matrix*tau,np.dot((np.array(p_matrix).T),np.linalg.inv(PriorViews+omega))),(q_matrix - np.dot(p_matrix,prior_ret)))
    #Calculation of Posterior Distribution (Variance)
    posterior_ret_distribution = cov_matrix + cov_matrix*tau - np.dot(np.dot(np.dot(cov_matrix*tau,np.array(p_matrix).T),np.linalg.inv(PriorViews+omega)), np.dot(p_matrix,cov_matrix*tau))
    print("\n Omega: \n",omega,"\n \n Diagonalised Omega: \n",diagnolised_omega,"\n\n Views: \n",PriorViews,"\n\n Posterior Return: \n",posterior_ret,"\n\n Posterior Return Distribution: \n",posterior_ret_distribution)
    #Unconstrained Weights for the final model
    opt_weight_unconstrained = np.dot(np.array(posterior_ret).T,np.linalg.inv(risk_aversion*posterior_ret_distribution))
    #Constrained weights limited to total sum = 1
    opt_weight_constrained =  opt_weight_unconstrained/sum(opt_weight_unconstrained)
    #Portfolio weights for US Equity, Foreign Equity and Emerging Equity
    final_weight_portfolio = {"US Equity": opt_weight_constrained[0],
                    "Foreign Equity": opt_weight_constrained[1],
                    "Emerging Equity": opt_weight_constrained[2]}
    #Portfolio Expected returns, std and sharpe
    portfolio_expected_ret = np.dot(opt_weight_constrained,posterior_ret)
    portfolio_var = np.dot(opt_weight_constrained,np.dot(posterior_ret_distribution,opt_weight_constrained))
    portfolio_sd = np.sqrt(portfolio_var)
    portfolio_sharpe = portfolio_expected_ret/portfolio_sd
    
    print("\n Constrained Weights of the Portfolio: \n",final_weight_portfolio,"\n\n Portfolio Expected Returns: ",portfolio_expected_ret,"\n Portfolio Std Deviation: ",portfolio_sd,"\n Portfolio Sharpe: ",portfolio_sharpe)

#%%
# =============================================================================
# Question 1 with suitable values
# =============================================================================
tau = 0.1
tau_v = 0.1
p_matrix = [[1,0,0],[0,1,-1]]
q_matrix = [0.015,0.03]

Black_Litterman(tau,tau_v,p_matrix,q_matrix)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Output of the Q1:
    
 1. Prior Market Weights: 
 [0.5, 0.4, 0.1] 

 2. Risk Aversion Parameter:  3 
 Scalar on Uncertainty in the Prior:  0.1 
 Scalar on Manager Views:  0.1

 3. Omega: 
 [[ 1.84309122e-04 -4.33080934e-05]
 [-4.33080934e-05  2.03531602e-04]] 
 
 4. Diagonalised Omega: 
 [[0.00018431 0.        ]
 [0.         0.00020353]] 

 5. Views: 
 [[ 1.84309122e-04 -4.33080934e-05]
 [-4.33080934e-05  2.03531602e-04]] 

 6. Posterior Return: 
 [ 0.01014154  0.01383731 -0.0005658 ] 

 7. Posterior Return Distribution: 
                  US Equity  Foreign EQ  Emerging EQ
    US Equity     0.001935    0.001672     0.002127
    Foreign EQ    0.001672    0.002598     0.002551
    Emerging EQ   0.002127    0.002551     0.004642

 8. Constrained Weights of the Portfolio: 
 {'US Equity': 0.8131853938849001, 'Foreign Equity': 1.2790914133227844, 'Emerging Equity': -1.0922768072076847} 

 9. Portfolio Expected Returns:  0.02656414116927188 
    Portfolio Std Deviation:  0.060325928788295714 
    Portfolio Sharpe:  0.44034367481509523
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%%
# =============================================================================
# Question 2 with suitable values
# =============================================================================
tau_v = 0.01
Black_Litterman(tau,tau_v,p_matrix,q_matrix)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
Output of the Q2:

1. Prior Market Weights: 
 [0.5, 0.4, 0.1] 

2. Risk Aversion Parameter:  3 
 Scalar on Uncertainty in the Prior:  0.1 
 Scalar on Manager Views:  0.01

3. Omega: 
 [[ 1.84309122e-05 -4.33080934e-06]
 [-4.33080934e-06  2.03531602e-05]] 
 
4. Diagonalised Omega: 
 [[1.84309122e-05 0.00000000e+00]
 [0.00000000e+00 2.03531602e-05]] 

5. Views: 
 [[ 1.84309122e-04 -4.33080934e-05]
 [-4.33080934e-05  2.03531602e-04]] 

6. Posterior Return: 
 [ 0.01411664  0.02023542 -0.00692878] 

7. Posterior Return Distribution: 
                  US Equity  Foreign EQ  Emerging EQ
    US Equity     0.001860    0.001607     0.002044
    Foreign EQ    0.001607    0.002538     0.002493
    Emerging EQ   0.002044    0.002493     0.004502

8. Constrained Weights of the Portfolio: 
 {'US Equity': 0.8801437117980625, 'Foreign Equity': 1.4670391384636037, 'Emerging Equity': -1.347182850261666} 

9. Portfolio Expected Returns:  0.05144516364057029 
 Portfolio Std Deviation:  0.06724397864438654 
 Portfolio Sharpe:  0.7650523463614967


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%%
# =============================================================================
# Question 3 with suitable values
# =============================================================================
tau_v = 0.1
p_matrix = [[0,0,1],[1,-1,0]]
q_matrix = [0.015,0.02]
Black_Litterman(tau,tau_v,p_matrix,q_matrix)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Output of the Q3:
    
1. Prior Market Weights: 
 [0.5, 0.4, 0.1] 

2. Risk Aversion Parameter:  3 
 Scalar on Uncertainty in the Prior:  0.1 
 Scalar on Manager Views:  0.1

3. Omega: 
 [[ 4.37484758e-04 -3.58387429e-05]
 [-3.58387429e-05  1.08649639e-04]] 
 
4. Diagonalised Omega: 
 [[0.00043748 0.        ]
 [0.         0.00010865]] 

5. Views: 
 [[ 4.37484758e-04 -3.58387429e-05]
 [-3.58387429e-05  1.08649639e-04]] 

6. Posterior Return: 
 [0.01129801 0.00166516 0.01110559] 

7. Posterior Return Distribution: 
                  US Equity  Foreign EQ  Emerging EQ
    US Equity     0.001972    0.001709     0.002127
    Foreign EQ    0.001709    0.002587     0.002503
    Emerging EQ   0.002127    0.002503     0.004594

8. Constrained Weights of the Portfolio: 
 {'US Equity': 2.477109557788812, 'Foreign Equity': -1.9236112624672697, 'Emerging Equity': 0.4465017046784575} 

 Portfolio Expected Returns:  0.029741946415246644 
 Portfolio Std Deviation:  0.08189498312410251 
 Portfolio Sharpe:  0.3631717753720776
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#%%

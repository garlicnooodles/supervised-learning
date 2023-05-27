#deleting objs
for name in dir():
    if not name.startswith('_'):
        del globals()[name]


#Importing all necessary libraries
import numpy as np
import pandas as pd 


#converting to numpy arrays
def to_numpy_arrays(X_pd,y_pd):
    X_np=X_pd.to_numpy()
    y_np=y_pd.to_numpy()
    return X_np,y_np

#calculating hypothesis
def hyp(X_hyp,theta_hyp,b_hyp):
    h=np.dot(X_hyp,theta_hyp)+b_hyp
    return h

# calculating the cost function
def cost_function(X_cf,y_cf,theta_cf,b_cf):
    m_cf,n_cf = X_cf.shape
    tv1=(y_cf-hyp(X_cf,theta_cf,b_cf))**2
    tv2=np.sum(tv1)
    J=tv2/(2*m_cf)
    return J

# computing the gradient for theta and b
def compute_gradient(X_gr,y_gr,theta_gr,b_gr):
    m_gr,n_gr = X_gr.shape
    dj_dw=np.zeros((n_gr,))
    dj_db=0
    for i in range(m_gr):
        err=(np.dot(X_gr[i],theta_gr)+b_gr)-y_gr[i]
        for j in range(n_gr):
            dj_dw[j]=(dj_dw[j]+err) * X_gr[i,j]
        dj_db=(dj_db+err)
    dj_dw=dj_dw/m_gr
    dj_db=dj_db/m_gr
    return dj_dw,dj_db

#gradient descent computation
def gradient_descent(X_gd,y_gd,theta_in_gd,b_in_gd,alpha_gd,epoch_gd):
    X_gd,y_gd=to_numpy_arrays(X_gd,y_gd)
    plot_cost=[]
    theta=theta_in_gd
    b=b_in_gd
    for i in range(epoch_gd):
        dj_dw,dj_db=compute_gradient(X_gd,y_gd,theta,b)
        theta=theta-alpha_gd*dj_dw
        b=b-alpha_gd*dj_db
        plot_cost.append(cost_function(X_gd,y_gd,theta,b))
        if i%(epoch_gd/10)==0:
            print(f"iteration {i:4d}: Cost {plot_cost[-1]:8.2f} ")
    return theta,b,plot_cost

#validation scores
def val_scores(X_vs,y_vs,theta_vs,b_vs):
    X_vs,y_vs=to_numpy_arrays(X_vs,y_vs)
    y_hat=hyp(X_vs,theta_vs,b_vs)
    J=cost_function(X_vs,y_vs,theta_vs,b_vs)
    return J

if __name__ == '__main__':
    print("Hola")
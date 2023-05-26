#Importing all necessary libraries
import numpy as np
import pandas as pd 

#converting to numpy arrays
def to_numpy_arrays(X,y):
    X=X.to_numpy()
    y=y.to_numpy()
    print(X.shape,y.shape)
    return X,y

#calculating hypothesis
def hyp(X,theta,b):
    h=np.dot(X,theta)+b
    return h

# calculating the cost function
def cost_function(X,theta,b,y):
    m,n = X.shape
    tv1=(y-hyp(X,theta,b))**2
    tv2=np.sum(tv1)
    J=tv2/(2*m)
    return J

# computing the gradient for theta and b
def compute_gradient(X,theta,b,y):
    m,n = X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        err=(np.dot(X[i],theta)+b)-y[i]
        dj_dw=(dj_dw+err) * X[i]
        dj_db=(dj_db+err)
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

#gradient descent computation
def gradient_descent(X,theta_in,b_in,y,alpha,epoch):
    X,y=to_numpy_arrays(X,y)
    plot_cost=[]
    theta=theta_in
    b=b_in
    for i in range(epoch):
        dj_dw,dj_db=compute_gradient(X,theta,b,y)
        theta=theta-alpha*dj_dw
        b=b-alpha*dj_db
        plot_cost.append(cost_function(X,theta,b,y))
        if i%(epoch/10)==0:
            print(f"iteration {i:4d}: Cost {plot_cost[-1]:8.2f} ")
    return theta,b,plot_cost

if __name__ == '__main__':
    print("Hola")
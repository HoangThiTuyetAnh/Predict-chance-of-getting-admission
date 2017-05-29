import numpy as np
import matplotlib.pyplot as plt
# supress scientific value
np.set_printoptions(suppress=True) 
data = np.loadtxt("ex2data1.txt",dtype=np.float64,delimiter=",")
# split data into features and outputs
X = data[::,0:2]
Y = data[::,-1:]

# Visualising the Dataset
X_get_admit = np.select([Y==1],[X])
X_not_admit = np.select([Y==0],[X])
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.xlim(30,110)
plt.ylim(30,110)
plt.scatter(X_get_admit[::,0:1],X_get_admit[::,-1:],label="Admitted",color="#2ecc71" )
plt.scatter(X_not_admit[::,0:1],X_not_admit[::,-1:],label="Not Admitted",color="#e74c3c")
plt.legend(loc = "upper right",frameon=True)
plt.show()

# sigmoid function
def sigmoid(Z):
    return (1/(1+np.exp(-Z)))

# hypothesis with sigmoid
def hypothesis(X_bias,Theta):
    #here Theta is 1d convert it into 2d
    hx = X_bias.dot((Theta.reshape((1,3))).transpose())
    return sigmoid(hx)

# cost function
def cost(Theta,X_bias,Y):
    m,n = X_bias.shape
    hx = (hypothesis(X_bias,Theta))
    first = np.multiply(-Y,np.log(hx))
    second = np.multiply((1-Y),np.log(1-hx))
    return (1.0/m)*np.sum(first - second)
m,n = X.shape
Theta = np.zeros(n+1)
X_bias = np.ones((m,n+1))
X_bias[::,1:] = X
print 'cost with zero initialize theta :'cost(Theta,X_bias,Y)

# gradient finding method used for advance optimization algorithm
def gradient(Theta,X_bias,Y):
    m,n = X_bias.shape
    grad = np.zeros(n) 
    hx = hypothesis(X_bias,Theta)
    error = hx -Y
    for i in xrange(n):
        element = np.multiply(error,X_bias[:,i].reshape((100,1)))
        grad[i] = np.sum(element)/len(X_bias)
    return grad    
print 'gradient with zero initialize theta'gradient(Theta, X_bias, Y)
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=Theta, fprime=gradient, args=(X_bias,Y))
print 'output of optimization: 'result
Theta = result[0]
print 'Final Theta: 'Theta # result is 1d array convert it to 2d array for bug free calculation
print 'Final shape of Theta'Theta.shape
print 'Cost of Final model'cost(Theta, X_bias, Y)
hx = hypothesis(X_bias,Theta)

# assume 1 if hx >= 0.5 and assume 0 if hx < 0.5
threshold = 0.5
prediction = np.select([hx >= threshold, hx < threshold],[1,0])
final_error =  (np.sum(np.absolute(Y-prediction))/len(X))*100
print 'accuracy', (100-final_error),'% with threshold ',threshold

# lets try with different threshold
threshold = 0.3
prediction = np.select([hx >= threshold, hx < threshold],[1,0])
final_error =  (np.sum(np.absolute(Y-prediction))/len(X))*100
print 'accuracy', (100-final_error),'% with threshold ',threshold
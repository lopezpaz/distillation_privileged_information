from keras.models import Sequential, Graph
from keras.objectives import categorical_crossentropy 
from keras.layers.core import Dense, Dropout, Activation
from sklearn.preprocessing import StandardScaler
import theano.tensor as T
import numpy as np
import theano

# experiment 1: noiseless labels as privileged info
def synthetic_01(a,n):
    x  = np.random.randn(n,a.size)
    e  = (np.random.randn(n))[:,np.newaxis]
    xs = np.dot(x,a)[:,np.newaxis]
    y  = ((xs+e) > 0).ravel()
    return (xs,x,y)

# experiment 2: noiseless inputs as privileged info (violates causal assump)
def synthetic_02(a,n):
    x  = np.random.randn(n,a.size)
    e  = np.random.randn(n,a.size)
    y  = (np.dot(x,a) > 0).ravel()
    xs = np.copy(x)
    x  = x+e
    return (xs,x,y)

# experiment 3: relevant inputs as privileged info
def synthetic_03(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    xs = xs[:,0:3]
    a  = a[0:3]
    y  = (np.dot(xs,a) > 0).ravel()
    return (xs,x,y)

# experiment 4: sample dependent relevant inputs as privileged info
def synthetic_04(a,n):
    x  = np.random.randn(n,a.size)
    xs = np.copy(x)
    #xs = np.sort(xs,axis=1)[:,::-1][:,0:3]
    xs = xs[:,np.random.permutation(a.size)[0:3]]
    a  = a[0:3]
    tt = np.median(np.dot(xs,a))
    y  = (np.dot(xs,a) > tt).ravel()
    return (xs,x,y)

def MLP(d,q):
    model = Sequential()
    model.add(Dense(q, input_dim=d))
    model.add(Activation('softmax'))
    model.compile('rmsprop','categorical_crossentropy')
    return model

def softmax(w, t = 1.0):
    e = np.exp(w / t)
    return e/np.sum(e,1)[:,np.newaxis]

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function

def distillation(d,q,t,l):
    graph = Graph()
    graph.add_input(name='x', input_shape=(d,))
    graph.add_node(Dense(q), name='w3', input='x')
    graph.add_node(Activation('softmax'), name='hard_softmax', input='w3')
    graph.add_node(Activation('softmax'), name='soft_softmax', input='w3')
    graph.add_output(name='hard', input='hard_softmax')
    graph.add_output(name='soft', input='soft_softmax')

    loss_hard = weighted_loss(categorical_crossentropy,1.-l)
    loss_soft = weighted_loss(categorical_crossentropy,t*t*l)

    graph.compile('rmsprop', {'hard':loss_hard, 'soft':loss_soft})
    return graph

def do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te):
    t = 1
    l = 1
    # scale stuff
    s_x   = StandardScaler().fit(x_tr)
    s_s   = StandardScaler().fit(xs_tr)
    x_tr  = s_x.transform(x_tr)
    x_te  = s_x.transform(x_te)
    xs_tr = s_s.transform(xs_tr)
    xs_te = s_s.transform(xs_te)
    y_tr  = y_tr*1.0
    y_te  = y_te*1.0
    y_tr  = np.vstack((y_tr==1,y_tr==0)).T
    y_te  = np.vstack((y_te==1,y_te==0)).T
    # privileged baseline
    mlp_priv = MLP(xs_tr.shape[1],2)
    mlp_priv.fit(xs_tr, y_tr, nb_epoch=1000, verbose=0)
    res_priv = np.mean(mlp_priv.predict_classes(xs_te,verbose=0)==np.argmax(y_te,1))
    # unprivivileged baseline
    mlp_reg = MLP(x_tr.shape[1],2)
    mlp_reg.fit(x_tr, y_tr, nb_epoch=1000, verbose=0)
    res_reg = np.mean(mlp_reg.predict_classes(x_te,verbose=0)==np.argmax(y_te,1))
    # distilled
    mlp_dist = distillation(x_tr.shape[1],2,t,l)
    soften = theano.function([mlp_priv.layers[0].input], mlp_priv.layers[0].get_output(train=False))
    p_tr   = softmax(soften(xs_tr.astype(np.float32)),t)
    mlp_dist.fit({'x':x_tr, 'hard':y_tr, 'soft':p_tr}, nb_epoch=1000, verbose=0)
    res_dis = np.mean(np.argmax(mlp_dist.predict({'x':x_te},verbose=0)['hard'],1)==np.argmax(y_te,1))
    return np.array([res_priv,res_reg,res_dis])

# experiment hyper-parameters
d      = 50
n_tr   = 200
n_te   = 1000
n_reps = 100
eid    = 0

np.random.seed(0)

# do all four experiments
for experiment in (synthetic_01, synthetic_02, synthetic_03, synthetic_04):
    eid += 1
    R = np.zeros((n_reps,3))
    for rep in xrange(n_reps):
        a   = np.random.randn(d)
        (xs_tr,x_tr,y_tr) = experiment(a,n=n_tr)
        (xs_te,x_te,y_te) = experiment(a,n=n_te)
        R[rep,:] += do_exp(x_tr,xs_tr,y_tr,x_te,xs_te,y_te)
    means = R.mean(axis=0).round(2)
    stds  = R.std(axis=0).round(2)
    print str(eid)+\
          ' '+str(means[0])+'\pm'+str(stds[0])+\
          ' '+str(means[1])+'\pm'+str(stds[1])+\
          ' '+str(means[2])+'\pm'+str(stds[2])

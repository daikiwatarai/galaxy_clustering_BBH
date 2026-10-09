"""Batched bivariate Poisson-lognormal (PLN) for (N_Q, N_SF) cylinder counts.

Every row carries its own mean nu (2,) and covariance Sigma (2,2); the lognormal covariance is
moment-matched, S_ij = ln[1 + (Sigma_ij - delta_ij nu_i) / (nu_i nu_j)].  The pmf is a
mode-centred adaptive Gauss-Hermite integral (20x20 nodes per row); a fixed grid is pointwise
wrong by up to ~8 nats.  Validated against PLN2 in sfq_cylinder_plnD_figures.py to 4e-5 nats.
nu and Xi are interpolated from a (ln R_perp, ln L_par) table built by fq_posterior_forecast.py.
"""
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.interpolate import RegularGridInterpolator
_z,_w=np.polynomial.hermite_e.hermegauss(20)
_Z1,_Z2=np.meshgrid(_z,_z,indexing="ij")
ZN=np.column_stack([_Z1.ravel(),_Z2.ravel()]); LW=np.log(np.outer(_w,_w).ravel())+0.5*(ZN**2).sum(1)

def S_of(nu,Sig):
    return np.log(1.0+(Sig-nu[:,:,None]*np.eye(2)[None])/(nu[:,:,None]*nu[:,None,:]))

def logpmf_batch(n,nu,S,iters=30,chunk=4000):
    n=np.asarray(n,float); out=np.empty(len(n))
    for k in range(0,len(n),chunk):
        sl=slice(k,min(k+chunk,len(n))); nn,vv,SS=n[sl],nu[sl],S[sl]
        Si=np.linalg.inv(SS); m=-0.5*np.stack([SS[:,0,0],SS[:,1,1]],1); ldS=np.linalg.slogdet(SS)[1]
        x=np.log((nn+0.5)/vv)
        for _ in range(iters):
            e=vv*np.exp(x)
            g=nn-e-np.einsum("mij,mj->mi",Si,x-m)
            H=-(e[:,:,None]*np.eye(2)[None]+Si)
            x=x-np.linalg.solve(H,g[:,:,None])[:,:,0]
        e=vv*np.exp(x)
        Cv=np.linalg.inv(e[:,:,None]*np.eye(2)[None]+Si); Lc=np.linalg.cholesky(Cv)
        X=x[:,None,:]+np.einsum("mij,gj->mgi",Lc,ZN)
        lam=vv[:,None,:]*np.exp(X)
        po=(nn[:,None,:]*np.log(lam)-lam-gammaln(nn[:,None,:]+1)).sum(2)
        dx=X-m[:,None,:]
        q=np.einsum("mgi,mij,mgj->mg",dx,Si,dx)
        lg=po-0.5*q-np.log(2*np.pi)-0.5*ldS[:,None]
        out[sl]=0.5*np.linalg.slogdet(Cv)[1]+logsumexp(lg+LW[None,:],axis=1)
    return out

def sample_batch(nu,S,rng):
    L=np.linalg.cholesky(S)
    x=-0.5*np.stack([S[:,0,0],S[:,1,1]],1)+np.einsum("mij,mj->mi",L,rng.standard_normal((len(nu),2)))
    return rng.poisson(nu*np.exp(x))

class Tables:
    def __init__(self,path):
        t=np.load(path); self.lR,self.lL=t["lR"],t["lL"]
        # the corner R<=0.038, L<=0.13 h^-1 Mpc has a negative cross term (power-law extrapolation
        # below the fit range).  No event with V>=0.05 Mpc^3 reaches it; floor it so the
        # log-interpolation cannot leak nan into neighbouring cells.
        mk=lambda A: RegularGridInterpolator((self.lR,self.lL),np.log(np.clip(A,1e-30,None)),
                                             bounds_error=False,fill_value=None)
        self.fq=[mk(t["NUQ"][:,:,i]) for i in (0,1)]
        self.fs=[mk(t["NUS"][:,:,i]) for i in (0,1)]
        self.fx=[mk(t["XI"][:,:,i]) for i in (0,1,2)]
    def __call__(self,R,L):
        p=np.column_stack([np.log(R),np.log(L)])
        p[:,0]=np.clip(p[:,0],self.lR[0],self.lR[-1]); p[:,1]=np.clip(p[:,1],self.lL[0],self.lL[-1])
        nuQ=np.column_stack([np.exp(f(p)) for f in self.fq])
        nuS=np.column_stack([np.exp(f(p)) for f in self.fs])
        x=[np.exp(f(p)) for f in self.fx]
        XI=np.empty((len(R),2,2)); XI[:,0,0]=x[0]; XI[:,0,1]=XI[:,1,0]=x[1]; XI[:,1,1]=x[2]
        return nuQ,nuS,XI

def sigmaD(nu,XI):
    return nu[:,:,None]*np.eye(2)[None]+nu[:,:,None]*nu[:,None,:]*XI

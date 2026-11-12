"""Phase8 weighted metric kernel; same two-column AUC and proper losses."""
from common_l2 import *
from scipy.special import logsumexp

def weighted(y,p,z,w):
 y=np.asarray(y,int);p=np.asarray(p,float);w=np.asarray(w,float);decision=p[:,1]>.5;n=w.sum(1);a=w[:,y==0].sum(1);b=w[:,y==1].sum(1)
 tn=w@((y==0)&~decision);tp=w@((y==1)&decision);fp=a-tn;fn=b-tp
 with np.errstate(invalid='ignore',divide='ignore'):
  result=np.stack([(tn+tp)/n,.5*(tn/a+tp/b),np.zeros(len(w)),.5*(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn)),tn/a,tp/b,np.zeros(len(w)),np.zeros(len(w)),np.zeros(len(w))],axis=1)
  # Match the project's macro two-probability-column AUROC, including ties.
  pos=y==1;neg=~pos
  def compare(x,v):return (x[:,None]>v[None,:]).astype(float)+.5*(x[:,None]==v[None,:])
  kernel=.5*(compare(p[pos,1],p[neg,1])+compare(p[neg,0],p[pos,0]).T)
  result[:,2]=np.sum((w[:,pos]@kernel)*w[:,neg],axis=1)/(a*b)
  loss=logsumexp(z.astype(float),1)-z[np.arange(len(y)),y] if len(z) else -np.log(p[np.arange(len(y)),y])
  finite=np.isfinite(loss);result[:,6]=(w[:,finite]@loss[finite])/n
  result[(w[:,~finite].sum(1)>0),6]=np.inf
  result[:,7]=(w@np.sum((p-np.eye(2)[y])**2,axis=1))/n
  confidence=p.max(1);binid=np.minimum((confidence*10).astype(int),9);correct=decision==y
  result[:,8]=sum(np.abs(w[:,binid==j]@(correct[binid==j]-confidence[binid==j])) for j in range(10))/n
 result[(a==0)|(b==0)]=np.nan
 return result

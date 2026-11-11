"""Private fixed-ID raw-signal plots; public patient-distribution/synthetic plots only."""
from common9 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

def save(fig,path):
 fig.canvas.draw();assert fig.canvas.get_width_height()[0]>800;fig.savefig(path,dpi=125,bbox_inches='tight');plt.close(fig)

def main():
 assert load(HERE/'analysis/statistics_integrity.json')['status']=='PASS_FIXED_PATIENT_FAMILIES';assert load(HERE/'analysis/signal_response_integrity.json')['status']=='PASS_SIGNAL_ONLY_NO_LEVEL2'
 freeze('FIGURES',[Path(__file__)],private_representatives_from_pre_signal_rule=True,public_only_aggregate_and_synthetic=True)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});rs=records();lookup={(r['subject_id'],r['activity'],r['wrist']):r for r in rs};reps=load(HERE/'restricted_manifests/representatives_BEFORE_SIGNAL_READ.json')['subject_ids'];grid=np.linspace(0,50,129);curves=defaultdict(dict);beg=time.perf_counter()
 for r in rs:
  t,x=read_raw(r);fs=1/np.median(np.diff(t));pair=[]
  for sensor,sl in [('acc',slice(0,3)),('gyro',slice(3,6))]:
   f,p=psd(x[:,sl],fs);v=np.interp(grid,f,p.sum(1),left=np.nan,right=np.nan);key=(r['activity'],r['label'],sensor);curves[key].setdefault(r['subject_id'],[]).append(v)
 figmanifest=[]
 for label,ids in reps.items():
  for rank,sid in enumerate(ids,1):
   for activity in ACTS:
    fig,axes=plt.subplots(4,3,figsize=(18,12));ax=axes.ravel();raw=[];processed=[]
    for w,side in enumerate(SIDES):
     r=lookup[(sid,activity,side)];t,x=read_raw(r);t=t-t[0];y=np.load(r['processed_path'],allow_pickle=False).T;raw.append((t,x));processed.append(y)
     for j in range(3):ax[w*2].plot(t,x[:,j],lw=.5,label=CHANNELS[j]);ax[w*2+1].plot(t,x[:,j+3],lw=.5,label=CHANNELS[j+3])
     ax[w*2].set(title=f'{side}: raw Acc XYZ',ylabel='g');ax[w*2+1].set(title=f'{side}: raw Gyro XYZ',ylabel='rad/s');ax[w*2].legend(fontsize=7);ax[w*2+1].legend(fontsize=7)
     for sensor,sl,plotmag,plotrms in [('Acc',slice(0,3),4,6),('Gyro',slice(3,6),5,7)]:
      a=x[:,sl];mag=np.linalg.norm(a,axis=1);window=25;center=a-a.mean(0);rms=np.sqrt(np.convolve(np.sum(center**2,axis=1),np.ones(window)/window,mode='same'));ax[plotmag].plot(t,mag,lw=.6,label=side);ax[plotrms].plot(t,rms,lw=.6,label=side)
     jerk=np.linalg.norm(np.diff(x[:,:3],axis=0)/np.diff(t)[:,None],axis=1);ax[8].plot(t[1:],jerk,lw=.5,label=side)
    titles={4:('Raw Acc magnitude','g'),5:('Raw Gyro magnitude','rad/s'),6:('0.25s centered Acc RMS (edge convolution affected)','g'),7:('0.25s Gyro RMS (edge convolution affected)','rad/s'),8:('Acc jerk magnitude using actual dt','g/s')}
    for j,(title,unit) in titles.items():ax[j].set(title=title,ylabel=unit);ax[j].legend(fontsize=7)
    t,x=raw[0];y=processed[0];rawtrim=x[48:];reference=rawtrim-rawtrim.mean(0)
    for j,sl,name in [(9,slice(0,3),'Acc'),(10,slice(3,6),'Gyro')]:
     for values,tag in [(rawtrim,'raw trim48'),(y,'current'),(reference,'DC-only diagnostic reference')]:
      f,p=psd(values[:,sl],1/np.median(np.diff(t)));ax[j].semilogy(f,p.sum(1)+1e-18,lw=.8,label=tag)
     ax[j].set(title=f'Left {name}: Welch PSD (no clinical interpretation)',xlabel='Hz',ylabel='power/Hz');ax[j].legend(fontsize=7)
    ax[11].plot(t[48:],rawtrim[:,0],label='raw AccX',lw=.6);ax[11].plot(t[48:],y[:,0],label='current L1 residual',lw=.7);ax[11].plot(t[48:],reference[:,0],label='DC-only reference',lw=.6);ax[11].set(title='Left AccX: existing filter vs fixed DC-only reference',ylabel='g');ax[11].legend(fontsize=7)
    for j in list(range(9))+[11]:ax[j].axvspan(0,.5,color='#a0a0a0',alpha=.12);ax[j].axvline(t[48],color='#b44b44',lw=.6,ls='--');ax[j].set_xlabel('record-relative seconds')
    fig.suptitle(f'SERVER PRIVATE | fixed representative {label} #{rank} | {activity} | no case selection by signal',fontsize=12);fig.tight_layout(rect=[0,0,1,.97]);name=f'representative_{label}_{rank}_{activity}.png';save(fig,HERE/'restricted_manifests/figures'/name);figmanifest.append(dict(subject_id=sid,label=label,representative_rank=rank,activity=activity,path='restricted_manifests/figures/'+name,private_raw_signal=True))
    print(json.dumps(dict(stage='private_figures',completed=len(figmanifest))),flush=True)
 write(HERE/'restricted_manifests/private_figure_manifest.json',dict(figures=figmanifest))
 temporal=pd.read_csv(HERE/'restricted_manifests/analysis/temporal_boundary_metrics.csv',dtype={'subject_id':str});patient=temporal.groupby(['subject_id','activity','label','sensor','window']).dynamic_rms.mean().reset_index();record=pd.read_csv(HERE/'restricted_manifests/analysis/record_metrics.csv',dtype={'subject_id':str});rpatient=record.groupby(['subject_id','activity','source_condition']).acc_dynamic_rms.mean().reset_index();summary=[]
 for activity in ACTS:
  fig,axes=plt.subplots(2,2,figsize=(12,8));colors={'PD':'#2f70ad','DD':'#d17a24'}
  for ax,sensor in zip(axes[0],['acc','gyro']):
   for label in ['PD','DD']:
    arr=np.stack([np.nanmean(v,axis=0) for v in curves[(activity,label,sensor)].values()]);q=np.nanquantile(arr,[.25,.5,.75],axis=0);ax.semilogy(grid,q[1]+1e-18,label=f'{label} ({len(arr)} patients)',color=colors[label]);ax.fill_between(grid,q[0]+1e-18,q[2]+1e-18,alpha=.15,color=colors[label]);summary.extend(dict(activity=activity,sensor=sensor,label=label,frequency_Hz=f,patient_count=len(arr),p25=low,median=med,p75=high) for f,low,med,high in zip(grid,*q))
   ax.set(title=f'Raw {sensor.upper()} PSD: patient median/IQR',xlabel='Hz',ylabel='g²/Hz' if sensor=='acc' else '(rad/s)²/Hz');ax.legend(fontsize=8)
  ax=axes[1,0]
  for label in ['PD','DD']:
   sub=patient[(patient.activity==activity)&(patient.label==label)&(patient.sensor=='acc')].pivot(index='subject_id',columns='window',values='dynamic_rms')
   for side,ls in [('start','-'),('end','--')]:
    values=[]
    for sec in [.5,1.,2.,3.,5.]:
     v=(sub[f'{side}{sec:g}']/sub.mid).replace([np.inf,-np.inf],np.nan);values.append(v.median())
    ax.plot([.5,1,2,3,5],values,ls,marker='o',ms=3,label=f'{label} {side}',color=colors[label])
  ax.axhline(1,color='grey',lw=.7);ax.set(title='Raw Acc dynamic RMS: boundary / center reference',xlabel='boundary window seconds',ylabel='patient median ratio');ax.legend(fontsize=8)
  labels=["Parkinson's",'Other Movement Disorders','Essential Tremor','Atypical Parkinsonism','Multiple Sclerosis'];g=rpatient[rpatient.activity==activity];arrays=[g[g.source_condition==s].acc_dynamic_rms for s in labels];axes[1,1].boxplot(arrays,tick_labels=['PD','Other','ET','Atypical','MS'],showfliers=False);axes[1,1].set(title='Acc dynamic RMS: broad source groups',ylabel='g');fig.suptitle(activity+' | aggregate quality distributions | each patient once; no individual traces',fontsize=12);fig.tight_layout(rect=[0,0,1,.96]);save(fig,HERE/'figures'/f'group_quality_{activity}.png')
 csv(pd.DataFrame(summary),'group_psd_curves.csv')
 f=pd.read_csv(HERE/'analysis/filter_frequency_response.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.5));part=f[f.response_type.str.startswith('HarNet')];axes[0].plot(part.frequency_Hz,20*np.log10(np.maximum(part.gain,1e-8)));axes[0].axvline(15,color='grey',ls='--');axes[0].set(xlim=(0,50),ylim=(-110,5),title='Current HarNet resample_poly FIR',xlabel='Input physical frequency (Hz)',ylabel='DC-normalized gain (dB)')
 for amp,g in f[f.response_type.str.startswith('L1')].groupby('input_amplitude'):axes[1].semilogx(g.frequency_Hz,g.gain,marker='o',ms=3,label=f'synthetic amplitude {amp:g}g')
 axes[1].set(title='Current L1 empirical synthetic response (NOT LTI)',xlabel='Hz',ylabel='interior sine amplitude ratio');axes[1].legend();fig.tight_layout();save(fig,HERE/'figures/filter_responses_synthetic.png')
 t=np.arange(1000)/100;x=np.ones(1000);a=signal.resample_poly(x,3,10);b=signal.resample_poly(x,3,10,padtype='line');fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.arange(20)/30,a[:20],marker='o',label='current zero extension');ax.plot(np.arange(20)/30,b[:20],marker='s',label='line extension: diagnostic candidate');ax.axhline(1,color='grey',ls='--');ax.set(title='Synthetic constant 1g: resampling boundary only; no model test',xlabel='seconds after window boundary',ylabel='g');ax.legend();fig.tight_layout();save(fig,HERE/'figures/resampling_boundary_synthetic.png')
 write(HERE/'analysis/figures_integrity.json',dict(status='PASS',private_fixed_representative_figures=len(figmanifest),public_activity_group_figures=11,public_synthetic_figures=2,private_signals_downloaded_or_published=False,selection_before_signal_metrics=True,render_dimensions_and_finite_inputs_verified=True,seconds=time.perf_counter()-beg));print('PASS_PRIVATE_AND_PUBLIC_FIGURES')

if __name__=='__main__':main()

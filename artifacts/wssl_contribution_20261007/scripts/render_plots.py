"""Standalone diagnostic bars: stdlib SVG, optional existing Pillow PNG runtime.

No fitted model, candidate/activity selection, external download or dependency
installation. Values come exclusively from the locked seed-first summary.
"""
import argparse
import csv
import html
import math
from pathlib import Path


def plot(root,metric,png=False):
    rows=[r for r in csv.DictReader((root/'analysis/contribution_summary_long.csv').open()) if r['kind']=='activity' and r['metric']==metric]
    names=['CrossArms','DrinkGlas','Entrainment','HoldWeight','LiftHold','PointFinger','Relaxed','RelaxedTask','StretchHold','TouchIndex','TouchNose']
    lookup={r['activity']:r for r in rows};assert set(lookup)==set(names)
    values=[tuple(float(lookup[a][k])*100 for k in ['mean_delta','ci_low','ci_high']) for a in names]
    W,H=2200,1000;L,R,T,B=160,80,160,210;pw=W-L-R;ph=H-T-B
    low=min(0,min(v[1] for v in values));high=max(0,max(v[2] for v in values));span=max(high-low,.1)
    low-=span*.12;high+=span*.12
    def y(value):return T+(high-value)/(high-low)*ph
    step=pw/11;width=step*.52
    title='冻结 WSSL：逐 activity '+('BA' if metric=='ba' else 'AUROC')+' contribution'
    subtitle='完整模型 − 仅关闭该 activity 的 WSSL residual；不关闭 STR activity token'
    footer='三个 seed 先聚合，15 个固定 development splits；95% bootstrap CI 仅作重复开发诊断。不能据图选择 activity。'
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         '<g font-family="PingFang SC,Microsoft YaHei,Noto Sans CJK SC,sans-serif" fill="#17324d">',
         f'<text x="{L}" y="58" font-size="36" font-weight="600">{html.escape(title)}</text>',
         f'<text x="{L}" y="103" font-size="23" fill="#526777">{html.escape(subtitle)}</text>',
         f'<text x="{L}" y="140" font-size="22">贡献（百分点）</text>']
    primitives=[]
    for j in range(6):
        v=low+(high-low)*j/5;yy=y(v)
        svg.append(f'<line x1="{L}" y1="{yy}" x2="{W-R}" y2="{yy}" stroke="#e6edf2"/>')
        svg.append(f'<text x="{L-20}" y="{yy+8}" text-anchor="end" font-size="22">{v:.2f}</text>')
        primitives.append(('tick',v,yy))
    svg.append(f'<line x1="{L}" y1="{y(0)}" x2="{W-R}" y2="{y(0)}" stroke="#637586" stroke-width="2"/>')
    for j,(a,(mean,lo,hi)) in enumerate(zip(names,values)):
        x=L+step*(j+.5);colour='#267eaa' if mean>=0 else '#cb7750';top=min(y(mean),y(0));height=max(1,abs(y(mean)-y(0)))
        svg.append(f'<rect x="{x-width/2}" y="{top}" width="{width}" height="{height}" fill="{colour}"/>')
        svg.append(f'<path d="M{x},{y(lo)} L{x},{y(hi)} M{x-12},{y(lo)} L{x+12},{y(lo)} M{x-12},{y(hi)} L{x+12},{y(hi)}" stroke="#203442" stroke-width="3" fill="none"/>')
        svg.append(f'<text x="{x}" y="{H-B+50}" text-anchor="middle" font-size="21">{a}</text>')
        svg.append(f'<text x="{x}" y="{H-B+87}" text-anchor="middle" font-size="21" fill="{colour}">{mean:+.3f}</text>')
        primitives.append(('bar',a,x,mean,lo,hi,colour))
    svg.append(f'<text x="{L}" y="{H-57}" font-size="23" fill="#526777">{html.escape(footer)}</text>')
    svg.append('</g></svg>');(root/'figures').mkdir(exist_ok=True)
    (root/'figures'/f'activity_{metric}_contribution.svg').write_text('\n'.join(svg)+'\n')
    if png:
        from PIL import Image,ImageDraw,ImageFont
        choices=[Path('/System/Library/Fonts/STHeiti Light.ttc'),Path('/System/Library/Fonts/PingFang.ttc'),Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')]
        font_path=next((p for p in choices if p.exists()),None)
        assert font_path,'An existing CJK font is required; do not silently replace Chinese labels'
        font=lambda n:ImageFont.truetype(str(font_path),n)
        im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
        d.text((L,22),title,font=font(36),fill='#17324d');d.text((L,74),subtitle,font=font(23),fill='#526777')
        d.text((L,119),'贡献（百分点）',font=font(22),fill='#17324d')
        for p in primitives:
            if p[0]=='tick':
                _,v,yy=p;d.line([(L,yy),(W-R,yy)],fill='#e6edf2',width=1)
                d.text((L-20,yy),f'{v:.2f}',anchor='rm',font=font(22),fill='#17324d')
        d.line([(L,y(0)),(W-R,y(0))],fill='#637586',width=2)
        for p in primitives:
            if p[0]!='bar':continue
            _,a,x,mean,lo,hi,colour=p
            d.rectangle((x-width/2,min(y(mean),y(0)),x+width/2,max(y(mean),y(0))),fill=colour)
            d.line([(x,y(lo)),(x,y(hi))],fill='#203442',width=3)
            for yy in [y(lo),y(hi)]:d.line([(x-12,yy),(x+12,yy)],fill='#203442',width=3)
            d.text((x,H-B+37),a,anchor='mt',font=font(21),fill='#17324d')
            d.text((x,H-B+74),f'{mean:+.3f}',anchor='mt',font=font(21),fill=colour)
        d.text((L,H-69),footer,font=font(23),fill='#526777')
        im.save(root/'figures'/f'activity_{metric}_contribution.png')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);ap.add_argument('--png',action='store_true');args=ap.parse_args()
    for metric in ['ba','auroc']:plot(args.root,metric,args.png)
    print('Two locked-result diagnostic bar plots exported; no dependency installation')

#!/usr/bin/env python3
"""
淘宝用户行为数据集 深度分析
数据: 318万条真实淘宝行为记录
字段: Id, user_id, age, gender, item_id, behavior_type, item_category, time, Province
behavior_type: 1=浏览, 2=收藏, 3=加购物车, 4=购买
"""
import pandas as pd, numpy as np, json, os
from collections import defaultdict

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'taobao_raw.txt')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_analysis_report.json')

def load():
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, skiprows=1,
        names=['Id','user_id','age','gender','item_id','behavior_type','item_category','time','Province'],
        dtype=str, encoding='latin1')
    for c in ['Id','user_id','age','gender','item_id','behavior_type']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Id','user_id','item_id','behavior_type'])
    df['behavior_type'] = df['behavior_type'].astype(int)
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['item_category'] = df['item_category'].astype(str).str.strip()
    print(f"Loaded {len(df):,} records")
    return df

def analyze(df):
    R = {}
    total = len(df)
    bmap = {1:'浏览(pv)', 2:'收藏(fav)', 3:'加购(cart)', 4:'购买(buy)'}
    
    # 1. 基本统计
    R['basic'] = {
        'total_records': total, 'unique_users': int(df['user_id'].nunique()),
        'unique_items': int(df['item_id'].nunique()),
        'unique_categories': int(df['item_category'].nunique()),
        'date_range': [str(df['time'].min()), str(df['time'].max())],
    }
    print(f"\n=== 基本统计 ===")
    for k,v in R['basic'].items(): print(f"  {k}: {v}")
    
    # 2. 行为分布
    bc = df['behavior_type'].value_counts().to_dict()
    R['behavior_distribution'] = {}
    print(f"\n=== 行为分布 ===")
    for bt in [1,2,3,4]:
        c = bc.get(bt,0)
        R['behavior_distribution'][bmap[bt]] = {'count':c, 'ratio':round(c/total*100,2)}
        print(f"  {bmap[bt]}: {c:,} ({c/total*100:.2f}%)")
    
    # 3. 用户序列长度分析 (核心!)
    us = df.groupby('user_id').size()
    R['user_sequence'] = {
        'mean': round(float(us.mean()),2), 'median': int(us.median()),
        'max': int(us.max()), 'min': int(us.min()),
        'std': round(float(us.std()),2),
        'p25': int(us.quantile(0.25)), 'p75': int(us.quantile(0.75)),
        'p90': int(us.quantile(0.90)), 'p95': int(us.quantile(0.95)),
        'p99': int(us.quantile(0.99)),
        'users_eq1': int((us==1).sum()), 'users_eq1_ratio': round(float((us==1).mean()*100),2),
        'users_ge3': int((us>=3).sum()), 'users_ge3_ratio': round(float((us>=3).mean()*100),2),
        'users_ge5': int((us>=5).sum()), 'users_ge5_ratio': round(float((us>=5).mean()*100),2),
        'users_ge10': int((us>=10).sum()), 'users_ge10_ratio': round(float((us>=10).mean()*100),2),
    }
    print(f"\n=== 用户序列长度 (关键!) ===")
    s = R['user_sequence']
    print(f"  mean={s['mean']}, median={s['median']}, max={s['max']}")
    print(f"  P25={s['p25']}, P75={s['p75']}, P90={s['p90']}, P95={s['p95']}, P99={s['p99']}")
    print(f"  仅1条: {s['users_eq1']:,} ({s['users_eq1_ratio']}%)")
    print(f"  >=3条: {s['users_ge3']:,} ({s['users_ge3_ratio']}%)")
    print(f"  >=5条: {s['users_ge5']:,} ({s['users_ge5_ratio']}%)")
    print(f"  >=10条: {s['users_ge10']:,} ({s['users_ge10_ratio']}%)")
    
    # 4. 多行为用户
    ub = df.groupby('user_id')['behavior_type'].nunique()
    R['multi_behavior_users'] = {
        'users_with_1_type': int((ub==1).sum()),
        'users_with_2_types': int((ub==2).sum()),
        'users_with_3_types': int((ub==3).sum()),
        'users_with_4_types': int((ub==4).sum()),
    }
    print(f"\n=== 多行为用户 ===")
    for k,v in R['multi_behavior_users'].items(): print(f"  {k}: {v:,}")
    
    # 5. 转化漏斗
    sets = {bt: set(df[df['behavior_type']==bt]['user_id']) for bt in [1,2,3,4]}
    isets = {bt: set(df[df['behavior_type']==bt]['item_id']) for bt in [1,2,3,4]}
    R['conversion'] = {
        'pv_to_fav': round(len(sets[1]&sets[2])/max(len(sets[1]),1)*100,2),
        'pv_to_cart': round(len(sets[1]&sets[3])/max(len(sets[1]),1)*100,2),
        'pv_to_buy': round(len(sets[1]&sets[4])/max(len(sets[1]),1)*100,2),
        'cart_to_buy': round(len(sets[3]&sets[4])/max(len(sets[3]),1)*100,2),
        'fav_to_buy': round(len(sets[2]&sets[4])/max(len(sets[2]),1)*100,2),
        'items_multi_behavior': int(len(isets[1] & (isets[2]|isets[3]|isets[4]))),
        'items_multi_behavior_ratio': round(len(isets[1]&(isets[2]|isets[3]|isets[4]))/max(len(isets[1]),1)*100,2),
    }
    print(f"\n=== 转化漏斗(用户级) ===")
    c = R['conversion']
    print(f"  浏览→收藏: {c['pv_to_fav']}%  浏览→加购: {c['pv_to_cart']}%  浏览→购买: {c['pv_to_buy']}%")
    print(f"  加购→购买: {c['cart_to_buy']}%  收藏→购买: {c['fav_to_buy']}%")
    print(f"  多行为商品: {c['items_multi_behavior']:,} ({c['items_multi_behavior_ratio']}%)")
    
    # 6. 商品长尾
    ic = df.groupby('item_id').size()
    ni = len(ic)
    R['item_longtail'] = {
        'items_1_interaction': int((ic==1).sum()),
        'items_1_ratio': round(float((ic==1).mean()*100),2),
        'top1pct_share': round(float(ic.nlargest(max(int(ni*0.01),1)).sum()/total*100),2),
        'top10pct_share': round(float(ic.nlargest(max(int(ni*0.10),1)).sum()/total*100),2),
    }
    print(f"\n=== 商品长尾 ===")
    il = R['item_longtail']
    print(f"  仅1次交互: {il['items_1_ratio']}%, Top1%贡献: {il['top1pct_share']}%, Top10%贡献: {il['top10pct_share']}%")
    
    # 7. user_id可靠性
    multi_u = us[us>1].index
    if len(multi_u) > 0:
        dm = df[df['user_id'].isin(multi_u)]
        ia = dm.groupby('user_id')['age'].nunique()
        ig = dm.groupby('user_id')['gender'].nunique()
        R['user_id_reliability'] = {
            'multi_record_users': int(len(multi_u)),
            'inconsistent_age': int((ia>1).sum()),
            'inconsistent_gender': int((ig>1).sum()),
        }
    else:
        R['user_id_reliability'] = {'multi_record_users':0, 'inconsistent_age':0, 'inconsistent_gender':0}
    print(f"\n=== User ID可靠性 ===")
    r = R['user_id_reliability']
    print(f"  多记录用户: {r['multi_record_users']:,}, 年龄不一致: {r['inconsistent_age']}, 性别不一致: {r['inconsistent_gender']}")
    
    # 8. DIN-V2训练适用性评估
    R['din_suitability'] = {
        'min_hist_len_3_eligible_users': int((us>=5).sum()),  # need 3 hist + val + test
        'min_hist_len_3_eligible_records': int(df[df['user_id'].isin(us[us>=5].index)].shape[0]),
        'strategy': ('user_id序列长度中位数=1，绝大多数用户仅1条记录。'
                     '需要按category+date构建session来获得有意义的行为序列，'
                     '或只使用>=5条记录的用户子集。'),
    }
    print(f"\n=== DIN-V2训练适用性 ===")
    d = R['din_suitability']
    print(f"  >=5条记录可用用户: {d['min_hist_len_3_eligible_users']:,}")
    print(f"  这些用户的记录总数: {d['min_hist_len_3_eligible_records']:,}")
    print(f"  策略: {d['strategy']}")
    
    return R

if __name__ == '__main__':
    df = load()
    R = analyze(df)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"\n分析报告已保存: {OUTPUT}")

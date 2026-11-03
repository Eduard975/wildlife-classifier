import csv, collections, sys
H="dataset_name,url_gcp,url_aws,url_azure,image_id,sequence_id,location_id,frame_num,original_label,scientific_name,common_name,datetime,annotation_level,kingdom,phylum,subphylum,superclass,class,subclass,infraclass,superorder,order,suborder,infraorder,superfamily,family,subfamily,tribe,genus,subgenus,species,subspecies,variety".split(',')
def grp(r):
    f,g,s=r['family'],r['genus'],r['scientific_name']
    if f=='ursidae': return 'bear'
    if s.startswith('canis lupus') and 'familiaris' not in s and 'dingo' not in s: return 'wolf'
    if g=='lynx': return 'lynx/bobcat'
    if s.startswith('puma concolor'): return 'puma'
    if g=='gulo': return 'wolverine'
    if s.startswith('sus scrofa'): return 'wild boar'
    if g=='panthera': return 'big cat'
    if g=='crocuta': return 'hyena'
    if s.startswith('canis latrans'): return 'coyote'
img=collections.Counter(); seq=collections.defaultdict(set); loc=collections.defaultdict(set); lvl=collections.defaultdict(collections.Counter)
for r in csv.DictReader(open('danger.csv'),fieldnames=H):
    g=grp(r)
    if not g: continue
    k=(g,r['dataset_name']); img[k]+=1; seq[k].add(r['sequence_id']); loc[k].add(r['location_id']); lvl[k][r['annotation_level']]+=1
tot=collections.Counter(); 
for (g,d),n in img.items(): tot[g]+=n
for g,n in tot.most_common():
    print(f"\n== {g}: {n} images")
    for (gg,d),m in sorted(img.items(),key=lambda x:-x[1]):
        if gg==g and m>=200: print(f"  {d:45s} img={m:7d} seq={len(seq[(gg,d)]):6d} loc={len(loc[(gg,d)]):4d} {dict(lvl[(gg,d)])}")

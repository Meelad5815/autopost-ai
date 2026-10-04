#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PAGES=[('Home','/','Hero, value proposition, services, trust, CTA'),('Services','/services/','Detailed services, deliverables, process and CTA'),('About','/about/','Business story, expertise, values and trust'),('Projects','/projects/','Portfolio/case studies and outcomes'),('Blog','/blog/','SEO articles and educational content'),('FAQ','/faq/','Common objections and answers'),('Contact','/contact/','WhatsApp/email/contact form and service area')]

def build(niche,business='MRK Digital',location=''):
    niche=niche.strip() or 'Web Development'; business=business.strip() or 'MRK Digital'; location=location.strip()
    return {'brand':business,'niche':niche,'location':location,'mode':'free-first','pages':[{'name':n,'path':p,'purpose':u} for n,p,u in PAGES],'seo':{'primary_keyword':niche+' services'+((' in '+location) if location else ''),'title_pattern':business+' | '+niche+' Services'+((' in '+location) if location else ''),'meta_description':business+' provides professional '+niche.lower()+' services with a clear process, practical solutions and direct contact.'},'ctas':['Get a Free Quote','WhatsApp Us','View Services','View Projects'],'deployment_targets':['GitHub Pages','Vercel','InfinityFree','WordPress'],'human_approval_required':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--niche',required=True); ap.add_argument('--business',default='MRK Digital'); ap.add_argument('--location',default=''); ap.add_argument('--out',default='data/website_factory'); a=ap.parse_args()
    spec=build(a.niche,a.business,a.location); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/'site_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    md='# '+spec['brand']+' — Website Factory\n\n**Niche:** '+spec['niche']+'\n**Location:** '+(spec['location'] or 'Not specified')+'\n\n## Pages\n'
    md+='\n'.join(['- **'+x['name']+'** — `'+x['path']+'` — '+x['purpose'] for x in spec['pages']])
    md+='\n\n## SEO\n- Primary keyword: '+spec['seo']['primary_keyword']+'\n- Title: '+spec['seo']['title_pattern']+'\n- Meta: '+spec['seo']['meta_description']+'\n\n## Deployment\n'+ '\n'.join(['- '+x for x in spec['deployment_targets']])+'\n\nHuman approval is required before publishing or changing a live site.\n'
    (out/'README.md').write_text(md,encoding='utf-8'); print(json.dumps(spec,ensure_ascii=False,indent=2))

if __name__=='__main__': main()

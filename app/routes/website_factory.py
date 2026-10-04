from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json, subprocess, sys
try:
    from app.auth import get_current_user
except Exception:
    def get_current_user(): return True
router=APIRouter(prefix='/api/website-factory',tags=['website-factory'])
class WebsiteRequest(BaseModel):
    niche:str
    business:str='MRK Digital'
    location:str=''
@router.post('/generate')
def generate(req:WebsiteRequest,user=Depends(get_current_user)):
    if not req.niche.strip(): raise HTTPException(400,'niche is required')
    cmd=[sys.executable,'website_factory.py','--niche',req.niche,'--business',req.business,'--location',req.location]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    if result.returncode!=0: raise HTTPException(500,result.stderr[-2000:])
    spec=Path('data/website_factory/site_spec.json')
    return {'ok':True,'artifact':str(spec),'spec':json.loads(spec.read_text(encoding='utf-8'))}

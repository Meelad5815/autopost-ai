#!/usr/bin/env python3
"""MRK deterministic, zero-cost website content generator.

Creates useful first-draft copy from a Website Factory spec without requiring
a paid AI API. Human review remains required before publishing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from html import escape

def generate(spec):
    brand=spec["brand"]; niche=spec["niche"]; location=spec.get("location","")
    area=f" in {location}" if location else ""
    return {
      "home":{"headline":f"{brand} — {niche} Solutions","intro":f"Practical, reliable {niche.lower()} services for clients{area}. We focus on clear communication, useful solutions and dependable support.","cta":"Request a Quote"},
      "services":[
        {"title":f"{niche} Consultation","body":"Understand your requirement, define the scope and recommend a practical solution."},
        {"title":f"{niche} Implementation","body":"Turn an approved requirement into a structured, usable solution with clear deliverables."},
        {"title":"Troubleshooting & Support","body":"Diagnose problems, explain the cause clearly and provide actionable next steps."},
        {"title":"Custom Projects","body":"Discuss your specific project and receive a tailored plan based on your goals and budget."}],
      "about":{"headline":f"About {brand}","body":f"{brand} provides {niche.lower()} solutions with an emphasis on practical delivery, transparent communication and long-term client value{area}."},
      "projects":{"headline":"Projects & Case Studies","body":"Add completed projects here with the problem, solution, technologies used and measurable result."},
      "blog":{"headline":"Guides & Insights","body":f"Publish original guides about {niche.lower()} to answer customer questions, demonstrate expertise and build organic search visibility."},
      "faq":[
        {"q":"How do I request a quote?","a":"Use the contact page and describe your requirement, preferred timeline and any important constraints."},
        {"q":"Can you work remotely?","a":"Yes, services that can be delivered online can be handled remotely."},
        {"q":"Can you build a custom solution?","a":"Yes. Share the requirement first so the scope and deliverables can be reviewed."}],
      "contact":{"headline":"Let's discuss your requirement","body":"Send your requirement, expected timeline and preferred contact method. Review all contact details before publishing."}
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--spec",default="data/website_factory/site_spec.json"); ap.add_argument("--out",default="data/website_factory/content.json"); a=ap.parse_args()
    spec=json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(generate(spec),ensure_ascii=False,indent=2),encoding="utf-8")
    print("Content generated:",out)

if __name__=="__main__": main()

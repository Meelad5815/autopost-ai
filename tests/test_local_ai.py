import os
from engine.ai import openai_json


def test_local_article_generation():
    os.environ["AI_PROVIDER"] = "local"
    os.environ["LOCAL_AI_LANGUAGE"] = "en"
    result = openai_json("", "local", """Topic: WordPress website development
Return strict JSON with keys:
- title
- meta_description
- excerpt
- content_html
- tags
- categories
- image_query
- faq_items
- related_keywords""", 10)
    assert result["title"]
    assert len(result["content_html"]) > 1500
    assert result["faq_items"]


def test_local_urdu_generation():
    os.environ["AI_PROVIDER"] = "local"
    os.environ["LOCAL_AI_LANGUAGE"] = "ur"
    result = openai_json("", "local", """Topic: WordPress
Return strict JSON with keys:
- title
- meta_description
- excerpt
- content_html
- tags
- categories
- image_query
- faq_items
- related_keywords""", 10)
    assert result["content_html"]

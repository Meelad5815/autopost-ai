import requests
from transformers import pipeline

# WordPress Settings
WP_URL = "https://dev-mr-kofficial786.pantheonsite.io/wp-json/wp/v2/posts"
USERNAME = "mrk786"
APP_PASSWORD = "wQdy juCz J5qa hRqq GlkT 3jiR"

# Custom AI Content Generator
generator = pipeline('text-generation', model='gpt2')  # local model
content = generator("Write a blog post about latest computer tech:", max_length=300)[0]['generated_text']

# Post to WordPress
data = {
    'title': "Latest Computer Tech Update",
    'content': content,
    'status': 'publish'
}
response = requests.post(WP_URL, auth=(USERNAME, APP_PASSWORD), json=data)
print(response.status_code, response.text)

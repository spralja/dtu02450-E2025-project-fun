# Generated by ChatGPT
import requests

url = "https://www.kaggle.com/api/v1/datasets/download/adityakadiwal/water-potability"  # Paste the copied link
output_path = "data/raw.zip"

response = requests.get(url, stream=True)

with open(output_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)

print("Download complete!")

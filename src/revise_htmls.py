# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import os

en_to_tr = {"Quick search": "Hızlı ara",
            "Table Of Contents": "İçindekiler",
            "Next": "Sonraki",
            "Previous": "Önceki",
            "Show Source": "Kaynak metni göster"}

def revise_html(html_file_path):
    with open(html_file_path) as hf:
        soup = BeautifulSoup(hf, 'html.parser')
        for el in soup.find_all(True):
            text = el.get_text()
            key_to_search = text.strip()
            if key_to_search in en_to_tr:
                new_text = text.replace(key_to_search,en_to_tr[key_to_search])
                el.string = new_text

        links = soup.find_all('a', class_="reference external")
        for link in links:
            link["target"] = "_blank"


    html = soup.prettify("utf-8")
    with open(html_file_path, "wb") as hf:
        hf.write(html)

INDEX_FILE_PATH = "_build/html/index.html"

revise_html(INDEX_FILE_PATH)



CHAPTERS_FOLDER_PATH = "_build/html/chapters"

for file_name in os.listdir(CHAPTERS_FOLDER_PATH):
    if file_name.endswith(".html"):
        html_file_path = "{}/{}".format(CHAPTERS_FOLDER_PATH, file_name)
        revise_html(html_file_path)

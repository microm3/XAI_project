import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://pokemondb.net/sprites"
SPRITES_DIR = "pokemon_sprites"

os.makedirs(SPRITES_DIR, exist_ok=True)

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def download_sprite(pokemon_name, img_url):
    if img_url.endswith('.gif') or '/shiny/' in img_url:
        print(f"Skipped: {img_url}")
        return

    folder = os.path.join(SPRITES_DIR, pokemon_name.lower())
    os.makedirs(folder, exist_ok=True)

    base_filename = os.path.basename(img_url)
    name_part, ext = os.path.splitext(base_filename)
    i = 0
    new_filename = base_filename
    while os.path.exists(os.path.join(folder, new_filename)):
        i += 1
        new_filename = f"{name_part}_{i}{ext}"

    filepath = os.path.join(folder, new_filename)
    img_data = requests.get(img_url).content
    with open(filepath, "wb") as f:
        f.write(img_data)
    print(f"Downloaded: {pokemon_name} -> {new_filename}")

def get_all_pokemon_sprite_pages():
    soup = get_soup(BASE_URL)
    links = soup.select("a[href^='/sprites/']")

    # Filter out only direct Pokémon pages like "/sprites/bulbasaur"
    sprite_pages = [
        urljoin(BASE_URL, a['href'])
        for a in links
        if a['href'].count('/') == 2 and not a['href'].endswith('/')
    ]
    return sorted(set(sprite_pages))

def scrape_main_normal_sprites():
    pages = get_all_pokemon_sprite_pages()
    for url in pages:
        pokemon_name = url.split("/")[-1]
        print(f"\nProcessing: {pokemon_name} -> {url}")
        soup = get_soup(url)

        table = soup.select_one("table.sprites-history-table")
        if not table:
            print(f"Skipped (no main table): {pokemon_name}")
            continue

        # Only look at the first <tr> (normal sprites)
        first_row = table.select_one("tbody > tr")
        if not first_row or "Normal" not in first_row.get_text():
            print(f"Skipped (no normal row): {pokemon_name}")
            continue

        # Download each image in that row
        for img_tag in first_row.select("img"):
            img_url = img_tag['src']
            download_sprite(pokemon_name, img_url)

if __name__ == "__main__":
    scrape_main_normal_sprites()

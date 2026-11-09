
import csv
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

BASE_URL = "https://www.rekrute.com/offres-emploi-maroc.html"

OUTPUT_DIR = Path("../data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "rekrute_raw.csv"

# Commencer avec 2 pages pour tester.
# Augmenter progressivement après vérification.
MAX_PAGES = 1000

WAIT_TIME = 45
DELAY_BETWEEN_JOBS = 1


FIELDS = [
    "job_id",
    "source",
    "title",
    "company",
    "location",
    "publication_date",
    "sector",
    "contract_type",
    "education_level",
    "experience",
    "remote",
    "description",
    "url",
    "scraped_at"
]


# ==========================================================
# 2. CONFIGURATION SELENIUM
# ==========================================================

def create_driver():

    options = Options()

    options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    driver.set_page_load_timeout(40)

    return driver


# ==========================================================
# 3. FONCTIONS UTILITAIRES
# ==========================================================

def generate_id(url):

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


def load_existing_urls():

    existing_urls = set()

    if OUTPUT_FILE.exists():

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                url = row.get("url")

                if url:
                    existing_urls.add(url)

    return existing_urls


def save_job(job):

    file_exists = OUTPUT_FILE.exists()

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(job)


def get_text(element, selector):

    try:

        return element.find_element(
            By.CSS_SELECTOR,
            selector
        ).text.strip()

    except Exception:

        return ""


def get_attribute(element, selector, attribute):

    try:

        return element.find_element(
            By.CSS_SELECTOR,
            selector
        ).get_attribute(attribute) or ""

    except Exception:

        return ""


# ==========================================================
# 4. EXTRACTION DES METADONNEES REKRUTE
# ==========================================================

def extract_rekrute_details(driver):

    details = {
        "sector": "",
        "contract_type": "",
        "education_level": "",
        "experience": "",
        "remote": ""
    }

    labels_mapping = {

        "secteur": "sector",

        "contrat": "contract_type",

        "niveau d'étude": "education_level",

        "expérience": "experience",

        "télétravail": "remote"
    }

    rows = driver.find_elements(
        By.CSS_SELECTOR,
        "div.jo-infoRow"
    )

    for row in rows:

        try:

            label = row.find_element(
                By.CSS_SELECTOR,
                "div.jo-infoLabel"
            ).text.strip()

            value = row.find_element(
                By.CSS_SELECTOR,
                "div.jo-infoValue"
            ).text.strip()

            field = labels_mapping.get(label.lower())

            if field:
                details[field] = value

        except Exception:

            continue

    return details


# ==========================================================
# 5. EXTRACTION DES LIENS DE LA PAGE
# ==========================================================

def extract_listing_links(driver):

    offers = driver.find_elements(
        By.CSS_SELECTOR,
        "li.post-id"
    )

    listings = []

    for offer in offers:

        try:

            title_element = offer.find_element(
                By.CSS_SELECTOR,
                "a.titreJob"
            )

            title_text = title_element.text.strip()

            url = title_element.get_attribute("href")

            if not url:
                continue

            url = urljoin(BASE_URL, url)

            # Extraction du titre et de la ville
            parts = title_text.split("|")

            title = parts[0].strip()

            location = ""

            if len(parts) > 1:

                location = parts[-1].strip(" )(")

            company = get_attribute(
                offer,
                "img.photo",
                "title"
            )

            publication_date = get_text(
                offer,
                "em.date span"
            )

            listings.append({

                "url": url,
                "title": title,
                "company": company,
                "location": location,
                "publication_date": publication_date

            })

        except Exception as error:

            print("Erreur lecture carte :", error)

    return listings


# ==========================================================
# 6. EXTRACTION DE LA FICHE DETAILLEE
# ==========================================================

def extract_job_details(driver, wait, listing):

    url = listing["url"]

    # Ouvrir une nouvelle fenêtre
    driver.execute_script(
        "window.open(arguments[0], '_blank');",
        url
    )

    driver.switch_to.window(
        driver.window_handles[-1]
    )

    try:

        wait.until(
            EC.presence_of_element_located(
                (By.TAG_NAME, "body")
            )
        )

        time.sleep(1)

        # Description : texte complet de la fiche.
        # Le nettoyage sera réalisé ultérieurement avec PySpark.
        description = driver.find_element(
            By.TAG_NAME,
            "body"
        ).text.strip()

        # Extraction des métadonnées avec les vrais
        # sélecteurs observés sur ta capture d'écran.
        details = extract_rekrute_details(driver)

        job = {

            "job_id": generate_id(url),

            "source": "ReKrute",

            "title": listing["title"],

            "company": listing["company"],

            "location": listing["location"],

            "publication_date": listing["publication_date"],

            "sector": details["sector"],

            "contract_type": details["contract_type"],

            "education_level": details["education_level"],

            "experience": details["experience"],

            "remote": details["remote"],

            "description": description,

            "url": url,

            "scraped_at": datetime.now(
                timezone.utc
            ).isoformat()

        }

        return job

    finally:

        # Toujours fermer la fenêtre de détail
        # et revenir à la page des résultats.
        if len(driver.window_handles) > 1:

            driver.close()

            driver.switch_to.window(
                driver.window_handles[0]
            )


# ==========================================================
# 7. SCRAPING PRINCIPAL
# ==========================================================

def scrape_rekrute():

    print("\n========== REKRUTE SCRAPING ==========\n")

    driver = create_driver()

    wait = WebDriverWait(driver, WAIT_TIME)

    seen_urls = load_existing_urls()

    print(
        "Nombre d'anciennes offres déjà enregistrées :",
        len(seen_urls)
    )

    total_new = 0

    try:

        for page in range(1, MAX_PAGES + 1):

            # Pagination conservée selon la logique
            # utilisée dans ton ancien scraper.
            page_url = f"{BASE_URL}?s=1&p={page}"

            print(f"\n========== PAGE {page} ==========")

            print("URL :", page_url)

            try:

                driver.get(page_url)

                wait.until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "li.post-id")
                    )
                )

            except TimeoutException:

                print(
                    "Aucune annonce détectée ou chargement expiré."
                )

                break

            listings = extract_listing_links(driver)

            if not listings:

                print("Aucun lien trouvé. Arrêt.")

                break

            print(
                "Nombre d'offres trouvées :",
                len(listings)
            )

            new_listings = [

                item for item in listings

                if item["url"] not in seen_urls

            ]

            if not new_listings:

                print(
                    "Toutes les offres de cette page "
                    "ont déjà été collectées."
                )

                continue

            print(
                "Nouvelles offres à traiter :",
                len(new_listings)
            )

            for listing in new_listings:

                # Ajouter l'URL immédiatement pour éviter
                # les répétitions pendant cette exécution.
                seen_urls.add(listing["url"])

                try:

                    job = extract_job_details(
                        driver,
                        wait,
                        listing
                    )

                    if not job["description"]:

                        print(
                            "Description vide :",
                            listing["url"]
                        )

                        continue

                    save_job(job)

                    total_new += 1

                    print(
                        f"Offre enregistrée : {total_new} "
                        f"| {job['title']}"
                    )

                    if not job["education_level"]:

                        print(
                            "Attention : niveau d'études absent."
                        )

                except Exception as error:

                    print(
                        "Erreur pendant l'extraction :",
                        listing["url"],
                        error
                    )

                    if len(driver.window_handles) > 1:

                        driver.close()

                        driver.switch_to.window(
                            driver.window_handles[0]
                        )

                time.sleep(DELAY_BETWEEN_JOBS)

            time.sleep(2)

    finally:

        driver.quit()

    print("\n========== RESULTAT FINAL ==========")

    print("Nouvelles offres collectées :", total_new)

    print("Total d'URL connues :", len(seen_urls))

    print("Fichier :", OUTPUT_FILE)


# ==========================================================
# 8. EXECUTION
# ==========================================================

if __name__ == "__main__":

    scrape_rekrute()

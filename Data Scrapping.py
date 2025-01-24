from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import json

url = "https://ec.europa.eu/sustainable-finance-taxonomy/faq"

# Path to local chrome driver
service = Service("/usr/local/bin/chromedriver")

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # run in headless mode (no browser UI)
driver = webdriver.Chrome(service=service, options=options)

try:
    # 1. Navigate to the FAQ page
    driver.get(url)

    # 2. Wait a bit for Angular to load/ render content
    time.sleep(5)  # Increase if needed

    # 3. Grab the final rendered HTML
    page_source = driver.page_source

finally:
    driver.quit()

# 4. Parse with BeautifulSoup
soup = BeautifulSoup(page_source, "html.parser")

# 5. Use the same selectors as you did in the browser dev tools
faq_divs = soup.select('div.container-question-item.full-width')

faqs = []
for block in faq_divs:
    # Extract question text
    question_el = block.select_one('span.ecl-accordion__toggle-title')
    if not question_el:
        continue
    question_text = question_el.get_text(strip=True)

    # Extract answer text (possibly multiple paragraphs)
    answer_el = block.select_one('div.ecl-accordion__content')
    if not answer_el:
        continue
    answer_text = answer_el.get_text("\n", strip=True)

    faqs.append({
        "question": question_text,
        "answer": answer_text
    })

# 6. Save to JSON
with open("faq_data.json", "w", encoding="utf-8") as f:
    json.dump(faqs, f, ensure_ascii=False, indent=2)

print(f"Scraped {len(faqs)} FAQs.")

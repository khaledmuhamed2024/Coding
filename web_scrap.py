import json
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Configure Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL of the speeches page
url = "https://www.beehive.govt.nz/portfolio/nationalactnew-zealand-first-coalition-government-2023-2026/foreign-affairs"
driver.get(url)

# Extract the speeches
speech_elements = driver.find_elements(By.CSS_SELECTOR, 'div.views-row')
speeches = []

for speech_element in speech_elements:
    title_element = speech_element.find_element(By.CSS_SELECTOR, 'h2.node-title a')
    title = title_element.text.strip()
    date = speech_element.find_element(By.CSS_SELECTOR, 'span.date-display-single').text.strip()
    speaker = speech_element.find_element(By.CSS_SELECTOR, 'span.field-content a').text.strip()
    portfolios = speech_element.find_element(By.CSS_SELECTOR, 'div.field-name-field-portfolio').text.strip()
    link = title_element.get_attribute('href')
    
    # Open the speech page
    driver.get(link)
    content = driver.find_element(By.CSS_SELECTOR, 'div.field-name-body').text.strip()
    
    # Create a speech dictionary
    speech_data = {
        'title': title,
        'date': date,
        'speaker': speaker,
        'portfolios': portfolios,
        'content': content
    }
    speeches.append(speech_data)
    
    # Save to JSON file
    file_name = f"{date} - {title}.json".replace('/', '-')
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(speech_data, f, ensure_ascii=False, indent=4)

    # Return to the main page
    driver.back()

# Cleanup
driver.quit()

print(f"Extracted {len(speeches)} speeches and saved them as JSON files.")

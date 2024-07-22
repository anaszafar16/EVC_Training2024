# this code was made by hussain from the IOT track 


import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL of the website
url = "https://www.bayut.sa/%D8%A7%D9%84%D8%B9%D9%82%D8%A7%D8%B1%D8%A7%D8%AA-%D9%84%D9%84%D8%A8%D9%8A%D8%B9/%D9%85%D9%86%D8%B7%D9%82%D8%A9-%D8%A7%D9%84%D8%B1%D9%8A%D8%A7%D8%B6/"

def extract_property_details(soup):
    properties = []
    listings = soup.find_all('article')  # Find all property listings
    for listing in listings:
        property_details = {}
        try:
            # Find the price element
            price = listing.find('span', attrs={'aria-label': 'Price'})
            if price:
                property_details['Price'] = price.text.strip()

            # Find the type element
            type_ = listing.find('span', attrs={'aria-label': 'Type'})
            if type_:
                property_details['Type'] = type_.text.strip()

            # Find the bedrooms element
            bedrooms = listing.find('span', attrs={'aria-label': 'Beds'})
            if bedrooms:
                property_details['Rooms'] = bedrooms.text.strip()

            # Find the bathrooms element
            bathrooms = listing.find('span', attrs={'aria-label': 'Baths'})
            if bathrooms:
                property_details['Toilets'] = bathrooms.text.strip()

            # Find the area element
            area = listing.find('h4')
            if area:
                property_details['Area'] = area.text.strip()

            if property_details:  # Ensure property_details is not empty
                properties.append(property_details)
        except AttributeError:
            continue
    return properties

def scrape_properties():
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        properties = extract_property_details(soup)
        return properties
    else:
        return []

def save_to_excel(properties, path):
    df = pd.DataFrame(properties)
    df.to_excel(path, index=False)

if __name__ == "__main__":
    properties = scrape_properties()
    if properties:
        save_to_excel(properties, 'C:/Users/hmaji/blud/housesWebScraping/properties.xlsx')
        print(f"Scraped {len(properties)} properties and saved to 'C:/Users/hmaji/blud/housesWebScraping/properties.xlsx'")
    else:
        print("Failed to scrape property details.")

# Cookie Clicker Bot

This project is a simple automation script to play the Cookie Clicker game using Selenium WebDriver. The script continuously clicks the big cookie and attempts to purchase available upgrades and items to maximize the cookie production.

## Features

- Automates the process of clicking the big cookie.
- Buys available upgrades and items to enhance cookie production.
- Prioritizes more expensive items when purchasing.

## Prerequisites

- Python 3.x
- Google Chrome browser
- ChromeDriver

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/cookie-clicker-bot.git
    cd cookie-clicker-bot
    ```

2. Install the required Python packages:

    ```bash
    pip install selenium webdriver-manager
    ```

## Usage

1. Run the script:

    ```bash
    python cookie_clicker_bot.py
    ```

2. The script will open the Cookie Clicker game in a new browser window and start clicking the big cookie, buying upgrades, and purchasing items automatically.

3. To stop the script, press `Ctrl+C` in the terminal.

## Code Overview

The script is structured as follows:

- **Setup WebDriver**: Initializes the Chrome WebDriver and opens the Cookie Clicker game.
- **click_cookie()**: Function to click the big cookie.
- **buy_upgrades()**: Function to buy available upgrades.
- **buy_items()**: Function to buy available items, prioritizing more expensive ones.
- **Main Loop**: Continuously clicks the cookie, buys upgrades, and purchases items with a short delay in between.

### Main Script

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up the webdriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open the Cookie Clicker game
driver.get("https://orteil.dashnet.org/cookieclicker/")

# Wait for the page to load
time.sleep(10)  # Adjust the sleep time if necessary

# Find the big cookie element
cookie = driver.find_element(By.ID, "bigCookie")

# keep clicking the cookie
def click_cookie():
    cookie.click()

# Functions for buying upgrades and items 
def buy_upgrades():
    upgrades = driver.find_elements(By.CSS_SELECTOR, ".upgrade.enabled")
    for upgrade in upgrades:
        try:
            upgrade.click()
        except:
            pass

def buy_items():
    items = driver.find_elements(By.CSS_SELECTOR, ".product.unlocked.enabled")
    for item in items[::-1]:  # Reverse to prioritize more expensive items
        try:
            item.click()
        except:
            pass

# Main loop to run the script
try:
    while True:
        click_cookie()
        buy_upgrades()
        buy_items()
        time.sleep(0.1)  # Adjust the sleep time if necessary
except KeyboardInterrupt:
    print("Script stopped by the user.")

# Close the browser after finishing
driver.quit()

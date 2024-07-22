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


# both functions are for buying upgrades and items 
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

# this is the main part to run the script
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

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# Function to scrape the Fear and Greed Index
def get_cfgi():
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Set up Chrome WebDriver
    service = Service("chromedriver.exe")  # Path to your ChromeDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # URL for the Fear and Greed Index
        url = "https://alternative.me/crypto/fear-and-greed-index/"
        driver.get(url)

        # Find the element containing the Fear and Greed Index value
        index_value_element = driver.find_element(By.CLASS_NAME, "fng-circle")
        index_value = index_value_element.text

        # Print the index value
        print(f"Fear and Greed Index: {index_value}")

        # Return the index value as an integer
        return int(index_value)
    except Exception as e:
        print(f"Error scraping Fear and Greed Index: {e}")
        return None
    finally:
        driver.quit()

# Test the function
if __name__ == "__main__":
    index = get_cfgi()
    if index is not None:
        print(f"Current Fear and Greed Index: {index}")

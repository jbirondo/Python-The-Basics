from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located

with webdriver.Firefox() as driver:
    wait = WebDriverWait(driver, 25)
    driver.get("https://ampleharvest.org/find-food/")
    driver.find_element(By.TAG_NAME, "input").send_keys("95112" + Keys.RETURN)
    first_result = wait.until(
        presence_of_element_located((By.ID, "ah_sidebar")))
    print(first_result)

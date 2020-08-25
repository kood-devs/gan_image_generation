"""
    Download images from flickr using selenium

"""
import os
import sys
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


def set_download_dir(download_path):
    # seleniumを使ったダウンロードファイルの保存先を設定
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_path}

    options.add_experimental_option("prefs", prefs)
    return options


def download_images_from_flickr(keyword, download_path=None, download_num=5, img_size='Small'):
    # ダウンロード先のパスを指定
    # download_path = str(download_path)
    # keyword = str(keyword)

    # 操作するブラウザを開く
    options = set_download_dir(download_path)
    driver = webdriver.Chrome('chromedriver.exe', options=options)

    # 操作するページを開く
    driver.get("https://www.flickr.com/")
    sleep(10)

    # 写真を検索
    search_engine = driver.find_element_by_id('search-field')
    search_engine.send_keys(keyword + Keys.ENTER)
    sleep(10)

    # ダウンロードを許可している写真であれば、ダウンロードを実施
    for i in range(int(download_num)):
        counter = 0
        get_images = driver.find_elements_by_class_name("overlay")

        while len(get_images) == 0 and counter < 30:
            get_images = driver.find_elements_by_class_name("overlay")
            counter += 1
            sleep(1)

        # ダウンロードを許可していないケースがあるのでtryで処理
        try:
            get_image = get_images[i]
            get_image.click()
            sleep(1)
        except:
            pass

        try:
            get_image = driver.find_element_by_class_name("ui-icon-download")
            get_image.click()
            sleep(1)
            get_image = driver.find_element_by_class_name(img_size)
            get_image.click()
            sleep(1)
        except:
            pass

        driver.back()  # 1枚ダウンロードしたら一覧表示に戻る
        sleep(5)

    driver.quit()


def main():
    # 条件を指定
    dir_path = os.getcwd() + '\\images'
    keyword = 'cat'
    download_num = 100

    # ダウンロードを実施
    download_images_from_flickr(keyword, dir_path, download_num)


if __name__ == '__main__':
    main()

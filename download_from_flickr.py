"""
    Download images from flickr using selenium
    参考：https://www.oreilly.co.jp/books/9784873117782/
"""
import os
import sys
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.keys import Keys


# ダウンロードしたい画像の条件を指定
keyword = 'cat'
dir_path = os.getcwd() + '\\images'
download_num = 100


def set_download_dir(download_path):
    # seleniumを使ったダウンロードファイルの保存先を設定
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_path}

    options.add_experimental_option("prefs", prefs)
    return options


def download_images_from_flickr(keyword, download_path=None, download_num=10, img_size='Small'):
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
    for i in range(download_num):
        counter = 0
        get_images = driver.find_elements_by_class_name("overlay")

        while len(get_images) == 0 and counter < 30:
            get_images = driver.find_elements_by_class_name("overlay")
            counter += 1
            sleep(1)

        # ダウンロードを許可していない画像があるのでtryで処理
        # 画像を選択
        try:
            get_image = get_images[i]
            get_image.click()
            sleep(1)
        except:
            pass

        # 選択した画像をダウンロード
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
    # ダウンロードを実施
    download_images_from_flickr(keyword, dir_path, download_num)


if __name__ == '__main__':
    main()

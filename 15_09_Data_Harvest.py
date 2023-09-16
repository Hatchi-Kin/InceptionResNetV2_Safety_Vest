from bs4 import BeautifulSoup
import requests
import os

def download_images_from_urls(urls, directory_path):
    try:
        page=0
        loop=0

        for URL in urls:
            page+=1
            print("Parsing : Page",page)

            getURL = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(getURL.text, 'html.parser')

            images = soup.find_all('img')
            resolvedURLs = []

            for image in images:
                src = image.get('src')
                resolvedURLs.append(requests.compat.urljoin(URL, src))

            os.makedirs(directory_path, exist_ok=True)

            for image in resolvedURLs:
                loop+=1
                webs = requests.get(image)
                file_path = os.path.join(directory_path, str(loop)+".jpg")
                print(file_path)
                with open(file_path, 'wb') as file:
                    file.write(webs.content)
    
    except Exception as e:
        print(f"An error occurred: {e}")




URLs_of_IMGs = []
for i in range(1, 10): 
    URL = f"https://www.gettyimages.fr/photos/webcam?assettype=image&numberofpeople=one&phrase=webcam&sort=mostpopular&license=rf,rm&page={i}"
    URLs_of_IMGs.append(URL)


print("\n SCRAPPING : \n")
download_images_from_urls(URLs_of_IMGs, "no_vests")

from bs4 import BeautifulSoup
import requests
import os



def download_images_from_urls(urls, directory_path):
    # Loop over each URL in the list
    for URL in urls:
        # Send a GET request to the URL with a User-Agent header
        getURL = requests.get(URL, headers={"User-Agent":"Mozilla/5.0"})
        # Parse the HTML content of the response using BeautifulSoup
        soup = BeautifulSoup(getURL.text, 'html.parser')
         
        # Find all the <img> tags in the HTML content
        images = soup.find_all('img')
        resolvedURLs = []
         
        # Sometimes, the src attribut of <img> tags don't contain the full URL of the image.
        # The `requests.compat.urljoin()` function is used to resolve the relative URL. 
        # It takes two arguments: the base URL of the page and the relative URL of the image. 
        # It returns the full URL of the image by combining the base URL and the relative URL.

        for image in images:
            src = image.get('src')
            resolvedURLs.append(requests.compat.urljoin(URL, src))
         
        # Create the directory to store the downloaded images
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Download each image and save it to the directory
        for image in resolvedURLs:
            webs = requests.get(image)
            # Extract the file name from the URL and remove any query string
            file_name = image.split('?')[0].split('/')[-1]
            # Write the image content to a file in the directory
            open(directory_path + '/' + file_name, 'wb').write(webs.content)




# URLs for "with_vest"
URLswith = []
for i in range(2, 21):
    URL = f"https://fr.freepik.com/search?demographic=number1&format=search&page={i}&people=include&query=safety+vest+yellow+vest+gilet+securit%C3%A9&type=photo"
    URLswith.append(URL)



# URLs for "without_vest"
URLswithout = []
for j in range(2, 21):
    URL = f"https://fr.freepik.com/search?ai=exclude&demographic=number1&format=search&page={j}&people=include&query=randon%C3%A9e+camping+marche+personne+seule&type=photo"
    URLswithout.append(URL)




download_images_from_urls(URLswithout, "without_vest")


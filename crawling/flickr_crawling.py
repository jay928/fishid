import flickrapi
import urllib.request
import os

keywords = ["background", "cock", "ostrich", "brambling", "vulture", 'cheetah', "lion", "window", "food", "hobby", "country", "mountain", "snow", "war", "human", "school", "wind"]

# Flickr api access key
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

for keyword in keywords:
    photos = flickr.walk(text=keyword,
                         tag_mode='all',
                         tags=keyword,
                         extras='url_c',
                         per_page=100,           # may be you can try different numbers..
                         sort='relevance')

    urls = []
    for i, photo in enumerate(photos):
        url = photo.get('url_c')
        if url is None:
            continue

        urls.append(url)
        print(keyword + ' url : ' + str(i))
        # get 50 urls
        if i > 1000:
            break

    print(keyword + 'total Size : ' + str(len(urls)))

    # Download image from the url and save it to '00001.jpg'

    dir_name = '/Volumes/SD/deeplearning/data/notfish/flickr/' + keyword + "/"
    os.mkdir(dir_name)

    for i, url in enumerate(urls):
        print(keyword + ' save : ' + str(i))
        try:
            urllib.request.urlretrieve(url, dir_name + str(i) + '.jpg')
        except:
            continue

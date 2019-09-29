import flickrapi
import urllib.request
import os

DIR = '/Volumes/SD/deeplearning/data/fish/'
keywords = ["bullethead parrotfish","bicolor parrotfish","clown triggerfish","pacific longnose parrotfish","bluebarred parrotfish","longnose emperor","convict tang","bluebanded surgeonfish","blackfin barracuda","great barracuda","oriental sweetlips","jewel damsel","spinecheek anemonefish","clark's anemonefish","false clown anemonefish","giant moray","sea cucumber","fire dartfish","yellowstripe goatfish","harlequin ghost pipefish","trumpetfish","cornetfish","mandarinfish","blotched porcupinefish","three spot dascyllus","wellowtail coris","longnose hawkfish","humbug dacylus","snowflake moray","bluegreen chromis","purple queen fish","redfin anthias","blackfin hogfish","squirrelfish","humphead bannerfish","pennant bannerfish","sharpnose puffer","blacktip reef shark","white margin unicornfish","blackspotted puffer","volitan lionfish","singular bannerfish","humphead warasse","squarespot anthias","half and half chromis","bird wrasse","humphead parrotfish","orangespine unicornfish","scribbled filefish","longnose filefish","white spotted eagle ray","flame angelfish","scissor tail sergeant","zebra pipefish","blue boxfish","staghorn damselfish","threespot angelfish","blue devil fish","coral grouper","lyer tail grouper","orangestriped triggerfish","circular spadefish","long fin spadefish","bluespot grouper","sergeant major fish","picasso triggerfish","shepard's angelfish","eastern triangular butterflyfish","majestic angelfish","emperor angelfish","regal angelfish","klein's butterflyfish","ornate butterflyfish","saddleback butterflyfish","pyramid butterflyfish","moorish idol","titan triggerfish","blackback butterflyfish","foxface rabbitfish","long nosed butterflyfish","raccoon butterflyfish","teardrop butterflyfish","theadfin butterflyfish","whale shark","mermaid"]

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

    dir_name = DIR + keyword + "/"
    os.mkdir(dir_name)

    for i, url in enumerate(urls):
        print(keyword + ' save : ' + str(i))
        try:
            urllib.request.urlretrieve(url, dir_name + str(i) + '.jpg')
        except:
            continue

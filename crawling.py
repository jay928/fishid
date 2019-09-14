from google_images_download import google_images_download
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def imageCrawling(keyword, dir) :
    response = google_images_download.googleimagesdownload()

    arguments = {"keywords":keyword,
                 "limit":100,
                 "print_urls":True,
                 "no_directory":True,
                 "output_directory":dir}
    paths = response.download(arguments)
    print(paths)
with open('C:\\workspace\\fishid\\fishid_github\\fishid\\data\\fish_list.txt', 'r') as f:
    for line in f:
        new_line = line.strip('\n')
        dir_path = "C:\\workspace\\fishid\img"
        dir_name = new_line
        os.mkdir(dir_path + "\\" + dir_name + "\\")
        dir = dir_path+"\\"+dir_name+"\\"
        print(new_line + " :: " + dir)

        imageCrawling(new_line, dir)
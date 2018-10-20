import jieba
import jieba.analyse
import numpy as np

from wordcloud import WordCloud
from PIL import Image
from os import path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# get data
def get_data(filename):
    with open(filename,encoding='utf-8') as f:
        content = f.read()
    return content

# create word-weight dictionary
def tokenize_content(content):
    jieba.analyse.set_stop_words('data/stop_words')
    tags = jieba.analyse.extract_tags(content, topK=50, withWeight=True)
    word_tokens_rank = dict()
    for tag in tags:
        word_tokens_rank[tag[0]] = tag[1]
    return word_tokens_rank

def create_mask(picturename):
    d = path.dirname(__file__)
    mask = np.array(Image.open(path.join(d, picturename)))
    return mask

# create the last effection demo
def gernerate_wordcloud(tags, mask):
    word_cloud = WordCloud(width=512, height=512, random_state=10,mask=mask,
                           background_color='white',font_path='/Users/xj2sgh/PycharmProjects/OneHourProgram/BasicPython/WordCloud/data/simsun.ttf')
    word_cloud.generate_from_frequencies(tags)
    plt.figure(figsize=(10,8), facecolor='white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == '__main__':
    filename = '/Users/xj2sgh/PycharmProjects/OneHourProgram/BasicPython/WordCloud/data/Personal.txt'
    picturename = '/Users/xj2sgh/PycharmProjects/OneHourProgram/BasicPython/WordCloud/data/picture.jpg'
    content = get_data(filename)
    word_tokens_rank = tokenize_content(content)
    mask = create_mask(picturename)
    gernerate_wordcloud(word_tokens_rank, mask)
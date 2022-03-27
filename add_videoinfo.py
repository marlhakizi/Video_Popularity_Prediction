
# Explore Youtube 8 Million dataset

## Retrieve video meta-data


import re
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import youtube_dl
# from IPython.display import YouTubeVide

# list all the tfrecord files for video-level training data
arr = os.listdir("./frame-level")
r = re.compile(".+\\.tfrecord$")
video_train = sorted(list(filter(r.match, arr)))
video_train = list(map(lambda orig_string: "./frame-level/" + orig_string, video_train))
record_num =list(map(lambda x: re.findall(r"\d+", x),video_train))
record_num = np.array(record_num).flatten()

import urllib.request
import re


# construct a URI like  URL data.yt8m.org/AB/ABCD.js
# map the pesudo random id to the real youtub id
def get_real_id(pesudo_id):
    url = "http://data.yt8m.org/2/j/i/{}/{}.js".format(pesudo_id[0:2], pesudo_id)
    response = urllib.request.urlopen(url).read().decode()
    real_id = response.split(",")[-1][1:-3]
    return real_id


# This function collects the data provided by youtube-dl based on the real Youtubeid
def get_video_metadata(real_id):
    url = "https://www.youtube.com/watch?v=" + real_id
    ydl = youtube_dl.YoutubeDL()
    result = ydl.extract_info(url, download=False)
    fields = [
        "title",
        "categories",
        "tags",
        "description",
        "is_live",
        "view_count",
        "like_count",
        "channel_url",
        "duration",
        "average_rating",
        "age_limit",
        "webpage_url",
    ]
    video_metadata = [result[field] for field in fields]
    return video_metadata

k = 0

# Create the pandas DataFrame
col_names = [ "title","categories","tags", "description","is_live","view_count","like_count",
    "channel_url","duration","average_rating","age_limit","webpage_url",
]

for k in range(10):
    video_train_i = video_train[k]


    # for video_train_i in video_train[:10]:
    puesdo_ids = []
    real_ids = []
    labels = []
    all_rgbs = []
    all_audios = []
    video_metadata_list = []

    for raw_record in tf.data.TFRecordDataset(video_train_i).take(3):
        tf_example = tf.train.SequenceExample()
        rt = tf_example.ParseFromString(raw_record.numpy())
        # exract features
        vid_id = (
            tf_example.context.feature["id"].bytes_list.value[0].decode(encoding="UTF-8")
        )
        puesdo_ids.append(vid_id)
        labels.append(tf_example.context.feature["labels"].int64_list.value)

        # extract rgb and audio feature at each frame

        ty = len(tf_example.feature_lists.feature_list["rgb"].feature)
        rgb = np.zeros((ty, 1024))
        audio = np.zeros((ty, 128))
        for i in range(ty):
            rgb[i] = tf.io.decode_raw(
                tf_example.feature_lists.feature_list["rgb"].feature[i].bytes_list.value[0],
                tf.uint8,
            )  # .numpy()

            audio[i] = tf.io.decode_raw(
                tf_example.feature_lists.feature_list["audio"]
                .feature[i]
                .bytes_list.value[0],
                tf.uint8,
            )  # .numpy()

        all_rgbs.append(rgb)
        all_audios.append(audio)

        # get video metadata
        try:
            real_id = get_real_id(vid_id)
            # Get the youtube-dl valuable metadata
            metadata = get_video_metadata(real_id)
        except:
            real_id = np.nan
            metadata = [np.nan for i in range(12)]

        real_ids.append(real_id)
        video_metadata_list.append(metadata)

        
    df = pd.DataFrame(video_metadata_list, columns=col_names)
    df.insert(0, "puesdo_id", puesdo_ids)
    df.insert(1, "real_id", real_ids)
    df["rgb_by_frame"] = all_rgbs
    df["audio_by_frame"] = all_audios
    df = df.fillna(value=np.nan)
    print(df.shape)

    df.to_csv("./frame-with-metadata/%srecord.csv"%record_num[k])

# df.head()


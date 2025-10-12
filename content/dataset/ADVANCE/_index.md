---
title: "<b>ADVANCE</b>: <b>A</b>u<b>D</b>io <b>V</b>isual <b>A</b>erial sce<b>N</b>e re<b>C</b>ognition datas<b>E</b>t"
summary: To our knowledge, the audiovisual aerial scene recognition task has not been explored before. For further facilitating the research in this field, we construct a new dataset, with high-quality images and scene labels, named as ADVANCE, which in summary contains 5075 pairs of aerial images and sounds, classified into 13 classes.
dataset: true
# Optional external URL for project (replaces project detail page).
external_link: https://zenodo.org/record/3828124
weight: 3
image:
  caption: 
  focal_point: Smart

people: ["Di Hu","Xuhong Li","Lichao Mou","Pu Jin","Dong Chen","Liping Jing","Xiao Xiang Zhu","Dejing Dou"]
people_links: ["https://dtaoo.github.io/","https://scholar.google.com.hk/citations?hl=en&user=giQLiacAAAAJ","https://scholar.google.com/citations?user=7k8GAaEAAAAJ&hl=zh-CN","https://www.asg.ed.tum.de/en/sipeo/team/pu-jin/","https://www.researchgate.net/profile/Dong-Chen-15","https://scholar.google.com.hk/citations?user=zStEDu4AAAAJ&hl=zh-CN","https://scholar.google.com/citations?user=CNakdIgAAAAJ&hl=en","https://scholar.google.ca/citations?user=qBHsQ04AAAAJ&hl=en"]

# institution: ["ruc","ruc","ruc","ruc"]
# institution_links: ["http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/"]

is_item: true
rating: 2021_1_1
---
AuDio Visual Aerial sceNe reCognition datasEt (ADVANCE) consists of 5,075 geotagged aerial imagesound pairs involving 13 scene classes. The audio data are collected from [Freesound](https://freesound.org/browse/geotags/), where we remove the audio recordings that are shorter than 2 seconds, and extend those that are between 2 and 10 seconds to longer than 10 seconds by replicating the audio content. From the location information, we can download the updated aerial images from [Google Earth](https://earthengine.google.com/). Finally, the paired data are labeled according to the annotations from [OpenStreetMap](https://www.openstreetmap.org/), also using the attached geographic coordinates from the audio recording. Note that, this dataset covers a large variety of scenes from across the world. For more details about this dataset, please refer to [our paper](https://arxiv.org/pdf/2005.08449.pdf). [Download this dataset here](https://zenodo.org/record/3828124)
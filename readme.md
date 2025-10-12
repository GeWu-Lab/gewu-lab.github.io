## 主页使用教程
由于时间仓促, 教程中可能有问题, 可以参照content目录下已有的项目仿照着写就好了  
### 第一步
1. clone本仓库, 然后按照下面的步骤修改
2. 修改完之后, 使用`hugo -d docs`生成文件到docs文件夹下(需要安装hugo, hugo的版本0.66.0, hugo的安装非常简单)  
3. push到github上, 等待几分钟, 就可以完成更新  
### 添加相册
1. 将图片添加到`static/img/photos`下(例如`WechatIMG241.jpeg`)  
2. 在`content/gallery/_index.md`中添加一个`[[item]]`, 格式如下:
```
[[item]]
    title = "图片的标题, 如果留空, 那么将不显示标题"
    content = "一句话简介图片, 如果留空, 那么将不显示内容"
    font_color = "#f8f8f2"  title和content的字体颜色
    align = "center"  # 标题和内容的对齐方式, 可用选项: ["center", "left", "right"].
    overlay_color = "#555"  # 图片不可用时, 会显示该颜色
    overlay_img = "photos/bWechatIMG241.jpeg"  # 图片的url(如果按照1去配置的话, 这里必须添加photos/的前缀, 如果放在了static/img/dir下的话, 可以添加dir/的前缀), "photos/WechatIMG241.jpeg"
    overlay_filter = 0.5  # rgba的alpha通道, 取值范围为[0, 1], 0表示不模糊, 1表示完全模糊(即全黑)
```

### 添加News
1. 在`content/home/news.md`中添加一个`[[item]]`, 格式如下:
```
[[item]]
    title = "新闻的主标题(例如: 发表论文: xxx同学论文被xxx会议录用)"
    subtitle = "新闻的副标题(例如: 论文题目)"
    subtitle_color = "#808080"  # 副标题的颜色(如果没用特殊需求, 不需要该这个)
    link = "http://aibox.ruc.edu.cn/paper.html"  # 主标题的链接
```

### 添加图标到主页:
1. 在`content/home/hero.md`中添加一个`[[cta]]`, 格式如下:
```
[[cta]]
  url = "http://www.baidu.com"  # 链接
  icon_pack = "fab"  # 图标库
  icon = "zhihu"  # 图标名称
```
这里的icon_pack和icon的使用方法详见`https://wowchemy.com/docs/getting-started/page-builder/` 中的icons一节, 下面是从中摘抄出来的一部分
> Icon pack “fab” includes the following brand icons:
> - twitter, weixin, weibo, linkedin, github, facebook, pinterest, twitch, youtube, instagram, soundcloud
> - [See all icons](https://fontawesome.com/icons?d=gallery&s=brands)  
> 
> Icon packs “fas” and “far” include the following general icons:
> - fax, envelope (for email), comments (for discussion forum)
> - [See all icons](https://fontawesome.com/icons?d=gallery&s=regular,solid)  
>   
> Icon pack “ai” includes the following academic icons:
> - cv, google-scholar, arxiv, orcid, researchgate, mendeley
> - [See all icons](https://jpswalsh.github.io/academicons/)
> - To enable the academic icon pack in v5+, set `ai = true` under `[icon.pack]` in `params.yaml`  
> 
> Icon pack “emoji” gives you the ability to use emojis as icons  
> - [See all icons](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md)  
> - Enter the emoji shortcode, such as `:heart:`, in Wowchemy’s `icon` field
> - Wowchemy v4.9+ is required to utilise the emoji icon pack and can currently only be used in the Featurette (skills) widget.
>  
> Icon pack “custom” gives you the ability to use custom SVG icons
> - Create an SVG icon in your favorite image editor or download one from a site such as [Flat Icon](https://www.flaticon.com/)
> - Place the custom SVG icon in `assets/media/icons/`, creating the folders if necessary
> - Reference the SVG icon name (without `.svg` extension) in the `icon` field
> - Wowchemy v4.9+ is required to utilise the custom icon pack and can currently only be used in the Featurette (skills) widget.

### 添加publication
1. 在`content/publication`下为新的publication创建一个文件夹(例如: `content/publication/paper1`)
2. 在`content/publication/paper1`下创建`cite.bib`, `featured.png`(or `featured.jpg`)和`index.md`  
3. `cite.bib`是paper的bibtex, `featured.png`(or `featured.jpg`)是一张在publication界面下展示的一张图片
4. `index.md`中的内容如下:
```
---
title: "这是paper的标题"  
authors:  # paper的作者, 
 - author1
 - author2
 - author3*  
publication: Transactions on Pattern Analysis and Machine Intelligence(TPAMI) 2021 # 发表在哪里  
publication_types: ["2"]  # 发表类型id, 和类型是一一对应  
publication_types_name: Journal Paper  # 类型
url_pdf: http://www.baidu.com  # pdf的链接(没有则不需要提供此项)
url_code: http://www.baidu.com  # code的链接(没有则不需要提供此项)
url_project: http://www.baidu.com  # project主页的链接(没有则不需要提供此项)
url_dataset: http://www.baidu.com  # dataset的链接(没有则不需要提供此项)
topic_types: ["2"]  # topic的类型id, 和topic的类型一一对应
topic_types_name: topic_scene_understanding  # topic的类型
rating : 2021_08_01  # 发表时间, 用于默认排序
---
```
> 发表类型id映射表
> Uncategorized = 0  
> Conference paper = 1  
> Journal paper = 2  
> Preprint = 3
> Report = 4  
> Book = 5
> Book section = 6
> Thesis = 7
> Patent = 8
> Workshop paper = 9  
> topic类型id映射表  
> Uncategorized = 0  
> Multi-Modal Learning Mechanism = 1
> Audio-Visual Learning = 2  
> Deep Learning with hashing = 3  

### 添加成员  
1. 在`content/authors`下创建一个文件夹, 名字为`年级_name`, 例如`content/authors/22_xxx`  
2. 在`content/authors/22_xxx`下创建`avatar.png`(or `avatar.jpg`)作为头像和`_index.md`  
3. `_index.md`中的内容如下:
```
---
name: xxx

superuser: true

role: Since 2022  # 这一行将显示在名字下面, 支持内嵌html标签以及emoji表情, 建议填写年级

website_url: http://www.kaito.org.cn/  # 可以配置个人主页, 如果有的话

user_groups:  # 可选项["Leader", "Grad Students", "Ph.D Students", "Master Students", "Alumni", "Undergraduate"]
- Undergraduate
---
个人简介, 控制在600-800个英文字符之内
```

### 添加project    
1. 在`content/project`中新建一个文件夹, 例如`content/project/xxx`
2. 在`content/project/xxx`下创建`featured.png`(or `featured.jpg`)和`_index.md`  
3. `featured.png`(or `featured.jpg`)是一张在project界面下展示的一张图片  
4. `_index.md`中的内容如下:  
```
---
title: "标题"  # 支持内嵌html, 例如使用<b>加粗文本, 案例请见`content/dataset/ADVANCE` 
summary: 一段简短的介绍, 将显示在project界面下

web_link: https://gewu-lab.github.io/CSOL_TPAMI2021/  # 如果不是使用内置的project主页, 可以配置该链接指向对应的链接

weight: 3  # 用于定义project的显示顺序, 数字越大, 越靠前
image:  # 暂时不需要使用它
  caption:
  focal_point: Smart
---
关于project的一段介绍(如果配置了web_link, 这里可以不写)
```


### 添加dataset
1. 在`content/dataset`中新建一个文件夹, 例如`content/dataset/xxx`
2. 在`content/dataset/xxx`下创建`featured.png`(or `featured.jpg`)和`_index.md`  
3. `featured.png`(or `featured.jpg`)是一张在dataset界面下展示的一张图片  
4. `_index.md`中的内容如下:  

```
---
title: "标题"  # 支持内嵌html, 例如使用<b>加粗文本, 案例请见`content/dataset/ADVANCE`  
summary: 一段简短的介绍, 将显示在dataset界面下  
dataset: true  # 必须配置为true
# Optional external URL for project (replaces project detail page).
web_link: https://gewu-lab.github.io/MUSIC-AVQA/  # 如果不是使用内置的dataset主页, 可以配置该链接指向对应的链接
external_link: https://zenodo.org/record/4079386#.X4PFodozbb2  # 数据集的下载链接, 该配置仅仅在不配置web_link中生效
image:
  caption: 
  focal_point: Smart

people: ["Di Hu","Xuhong Li","Lichao Mou","Pu Jin","Dong Chen","Liping Jing","Xiao Xiang Zhu","Dejing Dou"]
people_links: ["https://dtaoo.github.io/","https://scholar.google.com.hk/citations?hl=en&user=giQLiacAAAAJ","https://scholar.google.com/citations?user=7k8GAaEAAAAJ&hl=zh-CN","https://www.asg.ed.tum.de/en/sipeo/team/pu-jin/","https://www.researchgate.net/profile/Dong-Chen-15","https://scholar.google.com.hk/citations?user=zStEDu4AAAAJ&hl=zh-CN","https://scholar.google.com/citations?user=CNakdIgAAAAJ&hl=en","https://scholar.google.ca/citations?user=qBHsQ04AAAAJ&hl=en"]  # 作者和作者链接是一一对应的(如果配置了web_link, 这里可以不写)

# institution: ["ruc","ruc","ruc","ruc"]
# institution_links: ["http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/","http://ai.ruc.edu.cn/"]  # 机构和机构链接是一一对应的(如果配置了web_link, 这里可以不写)

is_item: true  # 是否显示作者, 机构以及他们的链接
rating: 2021_1_1
---
关于dataset的一段介绍(如果配置了web_link, 这里可以不写)
```
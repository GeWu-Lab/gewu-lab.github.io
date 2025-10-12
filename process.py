import os
import re
import shutil
import json
from bs4 import BeautifulSoup

def legal(s):
    pattern=r'[\\/:*?"<>|\r\n]+'
    return re.sub(pattern, "_", s)

publication_types = ["Uncategorized", "Conference Paper", "Journal Paper", "Preprint", "Report", "Book", "Book section", "Thesis", "Patent", "Workshop Paper"]

bs = BeautifulSoup(open("Publications.html", "r"), features="html.parser")
layout_content = bs.find(id="layout-content")

contents = {}
contents_meta = {}

cur_type = None

for sub in layout_content.children:
    if sub == "\n": continue
    if hasattr(sub,"attrs") and "id" in sub.attrs and (sub.attrs["id"] == "toptitle" or sub.attrs["id"] == "footer"): continue
    if sub.name == "h2":
        if sub.text not in contents.keys(): contents[sub.text] = []
        cur_type = sub.text
    else:
        contents[cur_type].append(sub)

for key in contents.keys():
    contents_meta[key] = []
    for sub in contents[key]:
        metadata = {
            "img": None,
            "pdf": None,
            "demo": None,
            "video": None,
            "code": None,
            "dataset": None,
            "title": None,
            "project": None,
            "conf": "",
            "author": ""
        }

        # 获取img_url
        img = sub.find_all("img")
        if img is not None:
            metadata["img"] = img[0].attrs["src"]

        # 填充其他的信息
        p = sub.find_all("p")[0]
        if p is not None:
            for link in p.contents:
                # print(link)
                if str(link).lower().find("pdf") != -1: meta = "pdf"
                elif str(link).lower().find("self-supervised learning for heterogeneous audiovisual scene analysis") != -1: meta = "pdf"
                elif str(link).lower().find("generalising combinatorial discriminant analysis through conditioning truncated rayleigh flow") != -1: meta = "pdf"
                elif str(link).lower().find("arxiv") != -1: meta = "pdf"
                elif str(link).lower().find("code") != -1: meta = "code"
                elif str(link).lower().find("demo") != -1: meta = "demo"
                elif str(link).lower().find("youtube") != -1: meta = "video"
                elif str(link).lower().find("dataset") != -1: meta = "dataset"
                elif str(link).lower().find("project") != -1: meta = "project"
                elif str(link).lower().find("cvpr") != -1: meta = "conf"
                elif str(link).lower().find("ieee") != -1: meta = "conf"
                elif str(link).lower().find("acm") != -1: meta = "conf"
                elif str(link).lower().find("eccv") != -1: meta = "conf"
                elif str(link).lower().find("iccv") != -1: meta = "conf"
                elif str(link).lower().find("tpami") != -1: meta = "conf"
                elif str(link).lower().find("aaai") != -1: meta = "conf"
                elif str(link).lower().find("oral presentation") != -1: meta = "conf"
                elif str(link).lower().find("neurips") != -1: meta = "conf"
                elif str(link).lower().find("icassp") != -1: meta = "conf"
                elif str(link).lower().find("knowledge and information systems") != -1: meta = "conf"
                elif str(link).lower().find("scientia sinica informations") != -1: meta = "conf"
                # elif str(link).lower().find(",") != -1: meta = "pass"
                elif str(link).lower() == " ": meta = "pass"
                elif str(link).lower() == "<br/>": meta = "pass"
                else: meta = "author"
                # meta1 = input(f"type(pass, {', '.join(list(metadata.keys()))}) > guess: [{meta}]")
                # if meta1 == "y" or meta1 == "":
                #     meta = meta
                # else: meta = meta1
                # print(f"meta: {meta}")
                # print()
                if meta == "pass": continue
                if meta == "conf":
                    if link.name == "b": metadata[meta] += f"<b>{link.text}</b>"
                    else: metadata[meta] += link.text

                elif meta == "author":
                    if link.name == "b": metadata[meta] += f"<b>{link.text}</b>"
                    else: metadata[meta] += link.text
                else:
                    try: metadata[meta] = link.attrs["href"]
                    except Exception as e:
                        print("!!!!!!!")
                        print(str(link))
                        print(f"error: {e}")
                    if meta == "pdf":
                        metadata["title"] = link.text
        contents_meta[key].append(metadata)
        
        pub_types = publication_types.index(key)
        for meta in contents_meta[key]:
            content  = "---" + "  \n"
            content += f"title: \"" + meta["title"] + "\"  \n"
            content += "authors:  \n"
            authors = meta["author"].split(", ")
            if "" in authors: authors.remove("")
            if " " in authors: authors.remove(" ")
            for author in authors:
                content += f" - {author}  \n"
            content += f"publication_types: [\"" + str(pub_types) + "\"]" + "  \n"
            content += f"publication: " + meta["conf"] + "  \n"
            content += "publication_types_name: " + key + "  \n"
            if meta["pdf"] is not None:
                content += "url_pdf: " + meta["pdf"] + "  \n"
            if meta["demo"] is not None:
                content += "url_demo: " + meta["demo"] + "  \n"
            if meta["video"] is not None:
                content += "url_video: " + meta["video"] + "  \n"
            if meta["code"] is not None:
                content += "url_code: " + meta["code"] + "  \n"
            if meta["dataset"] is not None:
                content += "url_dataset: " + meta["dataset"] + "  \n"
            if meta["project"] is not None:
                content += "url_project: " + meta["project"] + "  \n"
            content  += "---" + "  \n"
        os.makedirs(f"content/publication/{legal(meta['title'])}", exist_ok=True)
        
        if meta['img'].endswith("png") or meta["img"].endswith("PNG"): posifix = "png"
        elif meta["img"].endswith("jpg") or meta["img"].endswith("jpeg") or meta["img"].endswith("JPG"): posifix = "jpg"
        try: shutil.copy(meta['img'], f"content/publication/{legal(meta['title'])}/featured.{posifix}")
        except Exception as e:
            print(json.dumps(metadata, indent=4))
        with open(f"content/publication/{legal(meta['title'])}/index.md", "w") as fo: fo.writelines(content)
        
        # print("")
        # print("")
        # print("current paper results: ")
        # print(json.dumps(metadata, indent=4))
        # print("")
        # print("")


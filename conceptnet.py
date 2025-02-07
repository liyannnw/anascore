import requests
import json
import sys
from tqdm import tqdm 
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from requests.exceptions import RequestException
import os
import string  
from utils import tokenize

def read_argv():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("--concept_dict",action="store",
                        dest="concept_dict",help="To elaborate...")

    return parser.parse_args()


def fetch_json(url,retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            # response.raise_for_status()  # Raises an HTTPError for bad responses (4xx, 5xx)
            return response.json()
        except (ValueError, RequestException) as e:
            # print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return None
        except Exception as e:
            # print(f"Unexpected error: {e}")
            return None

def get_conceptnet(word):

    url = f"http://api.conceptnet.io/c/en/{word}"
    categories =dict()# set()
    # try:
    #     response = requests.get(url).json()
    # except ValueError as e:
    #     return None
    response = fetch_json(url)
    if response is None:
        return None
    else:
    
        edges = response.get('edges', [])
        
        for edge in edges:
            if 'rel' in edge and 'label' in edge['rel']:
                # categories.add(edge['rel']['label'])
                category = edge['rel']['label']
                if category not in categories: categories[category] = {"start":[],"end":[]}# set()
                for node in ["start", "end"]:
                    end_ = edge[node]
                    # labels=[]
                    # add_info=None
                    if "label" in end_:
                        add_label = end_["label"]
                        if "language" in end_:
                            add_lang=end_["language"]
                            add_info = f"{add_lang}: {add_label}"
                            # categories[category][node].add(f"{end_["language"]}: {end_["label"]}")
                            # labels.append(f"{end_["language"]}: {end_["label"]}")
                        else:
                            add_info = add_label
                            # labels.append(end_["label"])
                            
                        if add_info not in categories[category][node]:
                            categories[category][node].append(add_info)

                    if "sense_label" in end_:
                        sense_label= end_["sense_label"]
                        if (sense_label[-3:] !=" wn") and (sense_label not in stop_words) and len(sense_label)>1:
                            if sense_label not in categories[category][node]:
                                categories[category][node].append(sense_label)

                    # if "language" in end_ and end_["language"] == "en":
                    #     labels.append(end_["label"])
                   
                            
                    # if "sense_label" in end_:
                    #     sense_label= end_["sense_label"]
                    #     if sense_label[-3:] !=" wn":
                    #         labels.append(sense_label)

                    #     label = end_["sense_label"]
                    # else:
                    #     label = end_["label"]
                    # for l in labels:
                    #     if l not in categories[category][node]:
                    #         categories[category][node].append(l)

    

    
    return categories if categories else None


if __name__ == "__main__":
    args = read_argv()
    collected = set()

    if os.path.isfile(args.concept_dict):

        for line in open(args.concept_dict):
            json_line = json.loads(line.strip())
            for text, info in json_line.items():
                if text not in collected:
                    collected.add(text)

    
    with open(args.concept_dict, "a+") as f:

        for line in tqdm(sys.stdin):
            sents = line.strip().split("\t")
            for sent in sents:
                # chunks = [mwe] + non_literal_chunks.split("|||") + [literal_chunks.split("|||")[0]]
                words = [word for word in tokenize(sent) if  (word not in string.punctuation)] #(word not in stop_words) and
                add_lines=[]
                for word in set(words):
                    if word not in collected:
                        # phrase= chunk.replace(' ', '_')
                        word_concept_info = get_conceptnet(word)
                        if word_concept_info:
                            json_line={word:None}
                            json_line[word] = word_concept_info
                            # print(json.dumps(json_line, ensure_ascii=False), file=sys.stdout)
                            add_lines.append(json.dumps(json_line, ensure_ascii=False))
                            collected.add(word)


                for add_line in add_lines:
                    f.write(add_line + "\n")


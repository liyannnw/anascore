#!/usr/bin/env python
# coding=utf-8
import sys
import numpy as np


from tqdm import tqdm 
from utils import tokenize
from anascore import AnaScore
from formal_criteria import is_formal_analogy
from utils import load_conceptnet_dict, read_argv

"""
This script reads input analogies, classifies them based on formal and semantic criteria,
and computes an overall label and score for each analogy.
"""


verify_forms=[[0,1,2,3], # A : B :: C : D
              [0,2,1,3]] # A : C :: B : D



def overall(labels):

    def parse_label(label):
        label = label.split("-")
        return float(label[-1]) if label[0] == "semantic" else 0


    ABCD_label, ACBD_label = labels
    ABCD_label_ = ABCD_label.split("-")[0]
    ACBD_label_ = ACBD_label.split("-")[0]

    if ABCD_label_ == ACBD_label_:
        if ABCD_label_ == "no_criterion":
            overall_label = f"vague-{ABCD_label_}" 
        else:
            overall_label =f"evident-{ABCD_label_}" 
    else:
        overall_label = "-".join(["vague", [xxx for xxx in [ABCD_label_,ACBD_label_] if xxx !="no_criterion"][0]])

    


    if overall_label == "evident-formal":
        overall_score =1.0
    elif overall_label == "vague-no_criterion":
        overall_score = 0.0
    else:
        scores_ =[parse_label(label_) for label_ in verify_labels]
        overall_score = np.round(np.average(scores_),2)

    return overall_label,overall_score



    

if __name__ == "__main__":
    args = read_argv()

    # Initialize AnaScore with a concept dictionary
    concept_dict = load_conceptnet_dict(open(args.concept_dict))
    anascore= AnaScore(concept_dict=concept_dict) 
    
    for line in tqdm(sys.stdin):

        line = line.strip().split("\t")
        original_score = float(line[-1])

        # Convert all analogy terms (A, B, C, D) to lowercase for consistency
        nlg= [xx.lower() for xx in line[1:5]]

        verify_labels=[]
        for iform,form in enumerate(verify_forms):
            label=None

            quad = [nlg[idx] for idx in form]

            # Check if the analogy satisfies formal analogy criteria at the character level
            if is_formal_analogy(quad):
                label = "formal-char"

            # Check at the word level if not formal at the character level
            if not label:
                quad_word= [tokenize(sent) for sent in quad]

                if is_formal_analogy(quad_word):
                    label = "formal-word"

            # If not formal, evaluate using AnaScore for semantic similarity
            if not label:
                concept_info_quad=[]

                outputs= anascore.get_score_from_quad(quad)
                score = outputs.score

                # Assign a "semantic" label if the score is positive; otherwise, assign "no_criterion"
                label = f"semantic-{np.round(score,2)}" if score >0 else "no_criterion"



            verify_labels.append(label)
        
        # Compute the overall analogy label and score based on both verification results
        overall_label, overall_score = overall(verify_labels)


        line = line[:5]+verify_labels+[overall_label,str(overall_score)]
        print("\t".join(line), file=sys.stdout)

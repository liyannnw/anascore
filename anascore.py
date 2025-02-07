#!/usr/bin/env python
# coding=utf-8
from nltk.corpus import stopwords
from attrdict import AttrDict
import string
from conceptnet import get_conceptnet
from utils import tokenize, load_conceptnet_dict
from utils import read_argv



__verbose__=True


class AnaScore:
    def __init__(self, concept_dict=None,eliminate_stop_words=False):
        """
        Args:
            concept_dict (dict): A dictionary mapping words to ConceptNet information.
            eliminate_stop_words (bool): Whether to remove stop words from consideration.
        """
        self.concept_dict = concept_dict
        self.eliminate_stop_words= eliminate_stop_words

        # Load stop words and punctuation if elimination is enabled
        if eliminate_stop_words:
            self.stop_words = set(stopwords.words('english')) | set(string.punctuation)



    def conceptual_representation(self, txt,composition=True):
        """
        Generate a conceptual representation of a given text using ConceptNet.

        Args:
            txt (str): Input text (word or sentence).
            composition (bool): Whether to consider individual word concepts separately.

        Returns:
            dict: A mapping of relations to sets of associated concepts.
        """
        concepts = dict()

        def aggregate(concept_info):

            for rel,info in concept_info.items():
                if rel not in concepts: concepts[rel] = set()
                # relation_info[rel] |= set(list(itertools.chain(*info.values()))) #set(info["end"]) #

                # Include directional information in the concept relationships
                for dir, node_value in info.items():

                    concepts[rel] |= {":".join([dir,nv.split(": ")[1]]) for nv in node_value if nv.split(": ")[0] == "en"}


        for word in txt.split():
            if word in self.concept_dict: # 
                if self.eliminate_stop_words and word in self.stop_words:
                    pass
                else:
                    aggregate(self.concept_dict[word])
            else: 
                # Fetch ConceptNet data for unknown words and update the dictionary
                word_concept_info=get_conceptnet(word)
                if word_concept_info:
                    aggregate(word_concept_info)
                    self.concept_dict[word] = word_concept_info



        if not composition:
            if txt in self.concept_dict:
                aggregate(self.concept_dict[txt])


        return concepts
    
    def overlap(self,conceptA, conceptB):
        """
        Find common concepts shared between two concept representations.

        Args:
            conceptA (dict): Conceptual representation of term A.
            conceptB (dict): Conceptual representation of term B.

        Returns:
            dict: A dictionary mapping relations to shared concepts.
        """

        same_concept=dict()
        rels = set(conceptA.keys()) & set(conceptB.keys())
        for rel in rels:
            same_labels = set(conceptA[rel]) & set(conceptB[rel])
            
            if same_labels:
                same_concept[rel] = same_labels
        
        return same_concept
    

    def diff(self,conceptA, conceptB):
        """
        Compute the differences in concept mappings between two terms.

        Args:
            conceptA (dict): Conceptual representation of term A.
            conceptB (dict): Conceptual representation of term B.

        Returns:
            dict: A dictionary mapping relations to deletions and insertions.
        """

        diff_concept= dict()
        rels = set(conceptA.keys()) & set(conceptB.keys()) 

        for rel in rels:

            labelsA = set(conceptA[rel]) if rel in conceptA else set()
            labelsB = set(conceptB[rel]) if rel in conceptB else set()

            
            samelabels = labelsA & labelsB
            delete_labels= labelsA - samelabels
            insert_labels = labelsB - samelabels


            if delete_labels or insert_labels:
                diff_concept[rel] = {"delete": delete_labels, "insert": insert_labels}
        
        return diff_concept




    def same_in_diffs(self,left_ratio_diff, right_ratio_diff):
        """
        Identify transformations that match between two ratios.

        Args:
            left_ratio_diff (dict): Differences in A:B transformation.
            right_ratio_diff (dict): Differences in C:D transformation.

        Returns:
            dict: Matching deletions and insertions that establish parallelism.
        """
        rels = set(left_ratio_diff.keys()) & set(right_ratio_diff.keys())

        matches= dict()
        for rel in rels:

            delete_A = left_ratio_diff[rel]["delete"]
            delete_B = right_ratio_diff[rel]["delete"]
            same_deletion = delete_A & delete_B

            insert_A = left_ratio_diff[rel]["insert"]
            insert_B = right_ratio_diff[rel]["insert"]
            same_insertion = insert_A & insert_B



            is_rel_match=False

            if same_deletion and same_insertion:
                is_rel_match=True
                
            elif same_deletion and not same_insertion:
                if not insert_A and not insert_B:
                    is_rel_match=True

            elif same_insertion and not same_deletion:
                if not delete_A and not delete_B:
                    is_rel_match=True
            
            elif not (delete_A or delete_B or insert_A or insert_B):
                is_rel_match= True
            

            if is_rel_match:
                matches[rel] = {"delete": same_deletion, "insert": same_insertion}

        return matches
    
    def find_word_from_concept(self,rel_label= None,concept=None, words=None):
        """
        Find words from a given set that are linked to a specific concept under a given relational label.

        Args:
            rel_label (str): The relationship type in ConceptNet (e.g., 'RelatedTo', 'IsA').
            concept (set): A set of conceptual nodes to check for overlap.
            words (list): A list of words to search for relevant concepts.

        Returns:
            list: Words that are linked to the specified concept under the given relational label.
        """
        related_words=[]

        for word in words:
            if word in self.concept_dict and rel_label in self.concept_dict[word]:
                word_rel_concept=set()

                # Iterate through relation types and extract valid concept mappings
                for dir, node_value in self.concept_dict[word][rel_label].items():

                    word_rel_concept |= {":".join([dir,nv.split(": ")[1]]) for nv in node_value if nv.split(": ")[0] == "en"} 

                # If the extracted concept overlaps with the target concept, store the word
                if word_rel_concept & concept:
                    related_words.append(word)

        return related_words
    

    


    def content_involved(self,shared_concepts=None,edit_concepts=None,seq=None,rel=None):
        """
        Determine whether the content in a sentence is aligned with shared concepts.

        Args:
            shared_concepts (set): Concepts that remain unchanged across the analogy.
            edit_concepts (set): Concepts that change in the transformation.
            seq (str): The sentence being analyzed.
            rel (str): The relation under which transformations occur.

        Returns:
            bool: True if the shared concepts and edit concepts align in the sentence, False otherwise.
        """

        shared_words = self.find_word_from_concept(rel_label= rel,concept=shared_concepts, words=[w for w in seq.split()]) 
        edit_words = self.find_word_from_concept(rel_label= rel,concept=edit_concepts, words=[w for w in seq.split()]) 
        

        return set(shared_words) == set(edit_words)

            

        
    def parallelism(self,del_left=None,ins_left=None,
                    del_right=None,ins_right=None,
                    del_common=None,ins_common=None,
                    quad=None,rel=None):
        """
        Compute the degree of parallelism between two ratios.

        """

        pA = self.content_involved(shared_concepts=del_common,edit_concepts=del_left,seq=quad[0],rel=rel)
        pB = self.content_involved(shared_concepts=ins_common,edit_concepts=ins_left,seq=quad[1],rel=rel)

        pC = self.content_involved(shared_concepts=del_common,edit_concepts=del_right,seq=quad[2],rel=rel)
        pD = self.content_involved(shared_concepts=ins_common,edit_concepts=ins_right,seq=quad[3],rel=rel)


        # Compute the average parallelism score
        p=(pA+pB+pC+pD) //4

        return p

    def cover_words(self,concepts=None,seq=None,rel_parallelism=None,edit_oper=None):
        """
        Extract words from a sentence that correspond to covered conceptual transformations.

        Args:
            concepts (dict): Conceptual representations of the sentence.
            seq (str): The input sentence.
            rel_parallelism (dict): Mapping of relational types to parallelism scores.
            edit_oper (str, optional): Operation type ("delete" or "insert").

        Returns:
            list: Words in the sentence that correspond to conceptual transformations.
        """
        cov_ws=[]
        if edit_oper:
            for rel,diff in concepts.items():
                if rel in rel_parallelism and rel_parallelism[rel] ==1:
                    cov_ws += self.find_word_from_concept(rel_label= rel,concept=diff[edit_oper], words=seq.split())

        else:
            for rel, overlap in concepts.items():
                cov_ws += self.find_word_from_concept(rel_label= rel,concept=overlap, words=seq.split())


        return list(set(cov_ws))






    def non_overlap_quad(self,quad):
        """
        Compute the non-overlapping components of the given analogy.

        Args:
            quad (list): The four terms in the analogy (A, B, C, D).

        Returns:
            list: Sentences with overlapping words removed.
        """
        # quad_concept = [self.seq_concept_info(term,composition=True, eliminate_stop_words=False) for term in quad]
            

        # left_ratio_same = self.same_concept_labels(quad_concept[0], quad_concept[1])
        # right_ratio_same =  self.same_concept_labels(quad_concept[2], quad_concept[3])


        # cov_same_A = self.cover_words(concepts=left_ratio_same,seq=quad[0])
        # cov_same_B = self.cover_words(concepts=left_ratio_same,seq=quad[1])
        # cov_same_C = self.cover_words(concepts=right_ratio_same,seq=quad[2])
        # cov_same_D= self.cover_words(concepts=right_ratio_same,seq=quad[3])


        # quad_diff = [" ".join(list(set(sent.split())-set(matched_words))) for sent, matched_words in zip(quad,[cov_same_A,cov_same_B,cov_same_C,cov_same_D])]
        # quad_diff = [" ".join([word for word in term.split() if word not in string.punctuation]) for term in quad_diff]


        quad = [" ".join([word for word in term.split() if word not in string.punctuation]) for term in quad]

        overlap_AB = set(quad[0].split()) & set(quad[1].split())
        overlap_CD = set(quad[2].split()) & set(quad[3].split())

        quad_diff = [" ".join([word for word in term.split() if word not in overlap]) for term,overlap in zip(quad,[overlap_AB,overlap_AB,overlap_CD,overlap_CD])]


        return quad_diff


    def get_score_from_quad(self,quad):
        """
        Compute the AnaScore for a given analogy.

        Args:
            quad (list): A list containing four terms representing the analogy (A, B, C, D).

        Returns:
            AttrDict: A dictionary containing computed analogy scores and intermediate results.
        """
        outputs=dict()

        # Tokenize input terms
        quad =[" ".join(tokenize(term)) for term in quad]

        quad_non_overlap = self.non_overlap_quad(quad)
        outputs["quad_non_overlap"] = quad_non_overlap


        quad_diff_concept = [self.conceptual_representation(term) for term in quad_non_overlap]
        outputs["quad_concept"] = quad_diff_concept


        # Compute similarity and differences
        left_ratio_same = self.overlap(quad_diff_concept[0], quad_diff_concept[1])
        right_ratio_same =  self.overlap(quad_diff_concept[2], quad_diff_concept[3])
        outputs["left_ratio_same"] = left_ratio_same
        outputs["right_ratio_same"] = right_ratio_same

        left_ratio_diff = self.diff(quad_diff_concept[0], quad_diff_concept[1])
        right_ratio_diff = self.diff(quad_diff_concept[2], quad_diff_concept[3])
        outputs["left_ratio_diff"] = left_ratio_diff
        outputs["right_ratio_diff"] = right_ratio_diff



        rel_matches = self.same_in_diffs(left_ratio_diff,right_ratio_diff)
        outputs["relational_matches"] = rel_matches

        rel2para=dict()
        for rel in set(left_ratio_diff.keys()) & set(right_ratio_diff.keys()):
            rel2para[rel] = 0



        for rel, rel_sub in rel_matches.items():
            rel_p = self.parallelism(del_left=left_ratio_diff[rel]["delete"],ins_left=left_ratio_diff[rel]["insert"],del_right=right_ratio_diff[rel]["delete"],ins_right=right_ratio_diff[rel]["insert"],del_common=rel_sub["delete"],ins_common=rel_sub["insert"], quad=quad_non_overlap,rel=rel)
            rel2para[rel] = rel_p
        
        outputs["rel2parallelism"] = rel2para

        if max(len(left_ratio_diff),len(right_ratio_diff)) ==0:
            p_value=1.0
        else:
            p_value = sum(rel2para.values())/max(len(left_ratio_diff),len(right_ratio_diff))

        outputs["parallelism"] = p_value

        quad_covs=[{w:False for w in sent.split()} for sent in quad_non_overlap]




        cov_diff_A = self.cover_words(concepts=rel_matches,seq=quad_non_overlap[0],rel_parallelism=rel2para,edit_oper="delete")
        cov_diff_B = self.cover_words(concepts=rel_matches,seq=quad_non_overlap[1],rel_parallelism=rel2para,edit_oper="insert")
        
        cov_diff_C = self.cover_words(concepts=rel_matches,seq=quad_non_overlap[2],rel_parallelism=rel2para,edit_oper="delete")
        cov_diff_D = self.cover_words(concepts=rel_matches,seq=quad_non_overlap[3],rel_parallelism=rel2para,edit_oper="insert")

 

        cov_same_A = self.cover_words(concepts=left_ratio_same,seq=quad_non_overlap[0])
        cov_same_B = self.cover_words(concepts=left_ratio_same,seq=quad_non_overlap[1])
        cov_same_C = self.cover_words(concepts=right_ratio_same,seq=quad_non_overlap[2])
        cov_same_D= self.cover_words(concepts=right_ratio_same,seq=quad_non_overlap[3])



        covs_same=[cov_same_A,cov_same_B,cov_same_C,cov_same_D]
        covs_diff=[cov_diff_A,cov_diff_B,cov_diff_C,cov_diff_D]



        covs_in_quad=[set(same_) | set(diff_) for same_,diff_ in zip(covs_same,covs_diff)]

        for icov, cov_ws in enumerate(covs_in_quad):
            for w in cov_ws:
                if w in quad_covs[icov]: quad_covs[icov][w] = True



        cov_value = sum([sum(covv.values())/len(covv) if covv else 1 for covv in quad_covs])
        cov_value = cov_value/4
        outputs["content_coverage"] = cov_value

        score = p_value * cov_value
        outputs["score"] = score




        
        return AttrDict(outputs)
    






if __name__ =="__main__":
    args = read_argv()

    # Path to the prefetched ConceptNet dictionary
    concept_dict_fpath = args.concept_dict  if args.concept_dict else "data/word_concept_info.json" 

    # Load the conceptual information from file
    concept_dict = load_conceptnet_dict(open(concept_dict_fpath))

    # Initialize AnaScore with the loaded concept dictionary
    anascore= AnaScore(concept_dict=concept_dict)
    

    lines=[
            # ["How many plates do you want?",       "How many apples do you want?",   "How many pens do you have?",    "How many pencils do you have?"],
            # ["I do not want any problems.",     "I do not want a boyfriend.",      "I do not need any rest.", "I do not need a girlfriend."],
            ["king", "queen", "man","woman"],
           ] 

    for quad in lines:
        print(quad)
   
        
        outputs= anascore.get_score_from_quad(quad)


        # If verbose mode is enabled, print detailed output
        if __verbose__:
            for k,v in outputs.items():
                print(f"***** {k} *****")
                print(v)
                print()
        else:
            # Print only the final AnaScore value
            print(outputs.score)


export concept_info_fpath=data/concept_info_word_all_norm.json


for dataset in semantico mixgoogle mulnlis sats
do
    printf "*****${dataset}*****\n"
    export save_fpath=data/${dataset}-annotated.tsv

    cat data/${dataset}.tsv | python3 annotation.py \
    --concept_dict ${concept_info_fpath} \
    > ${save_fpath}


done

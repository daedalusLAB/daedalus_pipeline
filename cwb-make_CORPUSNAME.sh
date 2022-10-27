#!/bin/bash

data="/data/corpora/cqpweb/corpora/test"
regfile="test"
name="test"
registry="/data/corpora/cqpweb/registry"
infile="/tmp/corpus.txt"
threads="10"

export CORPUS_REGISTRY="$registry"

cwb-encode  -c utf8 -d $data -f $infile -R "$registry/$regfile" -xsB -P pos -P lemma -P lower -P prefix -P suffix -P is_digit -P like_num -P dep -P shape -P tag -P sentiment -P is_alpha -P is_stop -P head_text -P head_pos -P children -P startsecs -P startcentisecs -P endsecs -P endcentisecs -S s:0+id+text -S text:0+id+file -0 corpus

cwb-make "$name" word &
cwb-make "$name" pos &
cwb-make "$name" lemma &
cwb-make "$name" lower &
cwb-make "$name" prefix &
cwb-make "$name" suffix &
cwb-make "$name" is_digit &
cwb-make "$name" like_num &
cwb-make "$name" dep &
cwb-make "$name" shape &
cwb-make "$name" tag &
cwb-make "$name" sentiment &
cwb-make "$name" is_alpha &
cwb-make "$name" is_stop &
cwb-make "$name" head_text &
cwb-make "$name" head_pos &
cwb-make "$name" children &
cwb-make "$name" startsecs &
cwb-make "$name" startcentisecs &
cwb-make "$name" endsecs &
cwb-make "$name" endcentisecs &
wait




# cwb-encode -c utf8 -d $data -f $infile -R "$registry/$regfile" -xsB -P pos -P lemma -P wc -P lemma_wc -P orig -P ner -P normner -P tagsbefore/ \
#         -P tagsafter/ -P timex -P whatever1 -P whatever2 -P root -P indep/ -P out_acl/ -P out_advcl/ -P out_advmod/ -P out_amod/ -P out_appos/ -P out_aux/ -P out_auxpass/ -P out_case/ \
#         -P out_cc/ -P out_ccomp/ -P out_compound/ -P out_conj/ -P out_cop/ -P out_csubj/ -P out_csubjpass/ -P out_dep/ -P out_det/ -P out_discourse/ -P out_dobj/ \
#         -P out_expl/ -P out_iobj/ -P out_mark/ -P out_mwe/ -P out_neg/ -P out_nmod/ -P out_nsubj/ -P out_nsubjpass/ -P out_nummod/ -P out_parataxis/ -P out_punct/ \
#         -P out_ref/ -P out_root/ -P out_xcomp/ -P out_other/ -P aligned_word -P is_aligned -P startsecs -P startcentisecs -P endsecs -P endcentisecs \
#         -P duration -P phones/ -P phones_durations/ -P is_first_pass -P is_recognized \
#         -S s:0+id+text+reltime -S text:0+id+collection+file+date+year+month+day+time+duration+country+channel+title+video_resolution+video_resolution_original+scheduler_comment+language+recording_location+program_id+original_broadcast_date+original_broadcast_time+original_broadcast_timezone+local_broadcast_date+local_broadcast_time+local_broadcast_timezone+teletext_page \
#         -S turn:0 -S meta:0+type+description+value+originalvalue -S story:0 -S musicalnotes:0+value -0 corpus




# NICHT VERGESSEN, zus�tzlich zu den oben gelisteten muss noch "word" dazu.
#parallel -j $threads cwb-make "$name" ::: word pos lemma wc lemma_wc orig lower ner normner tagsbefore tagsafter timex root indep out_acl out_advcl out_advmod out_amod out_appos out_aux out_auxpass out_case out_cc out_ccomp out_compound out_conj out_cop out_csubj out_csubjpass out_dep out_det out_discourse out_dobj out_expl out_iobj out_mark out_mwe out_neg out_nmod out_nsubj out_nsubjpass out_nummod out_parataxis out_punct out_ref out_root out_xcomp out_other startsecs startcentisecs endsecs endcentisecs persononscreen speakeronscreen handmoving movingvertically movinghorizontally shouldermoving slidingwindow noslidingwindow notwithhead gestures timelinegestures timelinegestures_confidence

# cwb-make "$name" word &
# cwb-make "$name" pos &
# cwb-make "$name" lemma &
# cwb-make "$name" wc &
# cwb-make "$name" lemma_wc &
# cwb-make "$name" orig &
# # cwb-make "$name" lower &
# cwb-make "$name" ner &
# cwb-make "$name" normner &
# cwb-make "$name" tagsbefore &
# cwb-make "$name" tagsafter &
# cwb-make "$name" timex &
# cwb-make "$name" whatever1 &
# cwb-make "$name" whatever2 &
# cwb-make "$name" root &
# cwb-make "$name" indep &
# cwb-make "$name" out_acl &
# cwb-make "$name" out_advcl &
# cwb-make "$name" out_advmod &
# cwb-make "$name" out_amod &
# cwb-make "$name" out_appos &
# cwb-make "$name" out_aux &
# cwb-make "$name" out_auxpass &
# cwb-make "$name" out_case &
# cwb-make "$name" out_cc &
# cwb-make "$name" out_ccomp &
# cwb-make "$name" out_compound &
# cwb-make "$name" out_conj &
# cwb-make "$name" out_cop &
# cwb-make "$name" out_csubj &
# cwb-make "$name" out_csubjpass &
# cwb-make "$name" out_dep &
# cwb-make "$name" out_det &
# cwb-make "$name" out_discourse &
# cwb-make "$name" out_dobj &
# cwb-make "$name" out_expl &
# cwb-make "$name" out_iobj &
# cwb-make "$name" out_mark &
# cwb-make "$name" out_mwe &
# cwb-make "$name" out_neg &
# cwb-make "$name" out_nmod &
# cwb-make "$name" out_nsubj &
# cwb-make "$name" out_nsubjpass &
# cwb-make "$name" out_nummod &
# cwb-make "$name" out_parataxis &
# cwb-make "$name" out_punct &
# cwb-make "$name" out_ref &
# cwb-make "$name" out_root &
# cwb-make "$name" out_xcomp &
# cwb-make "$name" out_other &
# cwb-make "$name" aligned_word &
# cwb-make "$name" is_aligned &
# cwb-make "$name" startsecs &
# cwb-make "$name" startcentisecs &
# cwb-make "$name" endsecs &
# cwb-make "$name" endcentisecs &
# cwb-make "$name" duration &
# cwb-make "$name" phones &
# cwb-make "$name" phones_durations &
# cwb-make "$name" is_first_pass &
# cwb-make "$name" is_recognized &
# wait

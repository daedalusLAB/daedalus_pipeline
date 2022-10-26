#!/usr/bin/env python3

import whisper
import spacy
import argparse
import sys
# I have installed gentle in tools/gentle path. 
# If you have gentle installed in your system and is in your path you can remove this line
# If not, you can download it from https://lowerquality.com/gentle/ and install it
sys.path.append('tools/gentle')
import subprocess
import gentle
from gentle import resampled, standard_kaldi, Resources
from gentle.forced_aligner import ForcedAligner


# Get and audio/video file and a transcription. Align the audio/video with the transcription
# and return gentle's result
def align_audio_text(audio_file, transcription):

  disfluencies = set(['uh', 'um'])
  resources = gentle.Resources()

  with gentle.resampled(audio_file) as wavfile:
      print("starting alignment")
      aligner = gentle.ForcedAligner(resources, transcription, nthreads=4, disfluency=False, conservative=False, disfluencies=False)
      result = aligner.transcribe(wavfile)

  return result


# Get gentle's result and return a list of words with their start and end timestamps
def get_word_timestamps(result):
    word_timestamps = []
    prev_start = 0 
    prev_end = 0
    for word in result.words:
        if word.start is None:
            word.start = prev_start        
        if word.end is None:
            word.end = prev_end
        word_timestamps.append({'word': word.word, 'start': "{:.2f}".format(word.start), 'end': "{:.2f}".format(word.end)})
        prev_start = word.start
        prev_end = word.end

    return word_timestamps


# Get an audio/video file and generate a transcription using whisper
# then align the audio with the transcription using gentle.
# Process transcription with spacy to get NLP features
# and finally generate a vrt file with the name of the audio/video file + .vrt
def generate_vrt(file):

  print('Loading whisper model...')
  model = whisper.load_model('large')

  print('Transcribing audio...')
  results = model.transcribe(file, max_initial_timestamp=None)

  aligned_results = align_audio_text(file,results['text'])
  word_timestamps = get_word_timestamps(aligned_results) 

  print('Loading spacy model...')
  nlp = spacy.load('en_core_web_lg')
  doc = nlp(results['text'])
  sentencizer = nlp.add_pipe("sentencizer")

  vrt_file = open(file + '.vrt', 'w')
  vrt_file.write('Prueba\n')

  print('Writing to vrt file...')
  last_start = 0
  last_end = 0
  for sent in doc.sents:
      vrt_file.write("<s>\n")
      for token in sent:
        # find the word timestamp for the token
        found = False
        for word in word_timestamps:
          # if word starts with token print token and timestamp
          if word['word'].startswith(token.text):
            found = True
            last_start = word['start']
            last_end = word['end']
            vrt_file.write(token.text + " | " + token.lemma_ + " | " +  
                          token.pos_ + " | " +  token.tag_ + " | " +  token.dep_ + " | " +  token.shape_ + " | " +  
                          str(token.is_alpha) + " | " +  str(token.is_stop) + " | " +  token.head.text + " | " +  
                          token.head.pos_ + " | " +  str([child for child in token.children]) + " | " + 
                          word['start'] + " | " +  word['end'] + "\n" )  
            # delete all words until the used one
            word_timestamps = word_timestamps[word_timestamps.index(word)+1:]
            break
        if not found:
          vrt_file.write(token.text + " | " + token.lemma_ + " | " +  
                          token.pos_ + " | " +  token.tag_ + " | " +  token.dep_ + " | " +  token.shape_ + " | " +  
                          str(token.is_alpha) + " | " +  str(token.is_stop) + " | " +  token.head.text + " | " +  
                          token.head.pos_ + " | " +  str([child for child in token.children]) + " | " + 
                          str(last_start) + " | " + str(last_end) + "\n") 
      vrt_file.write("</s>\n")
  vrt_file.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--file', type=str, help='audio or video file')
  args = parser.parse_args()

  generate_vrt(args.file)

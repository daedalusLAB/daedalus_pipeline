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


def get_secs(time):
  # if time has a dot, return the seconds part
  if '.' in str(time):
    return str(time).split('.')[0]
  else: 
    return str(time)

def get_msecs(time):
  # if time has a dot, return the milliseconds part
  if '.' in str(time):
    return str(time).split('.')[1]
  else:
    return '00'

# Get an audio/video file and generate a transcription using whisper
# then align the audio with the transcription using gentle.
# Process transcription with spacy to get NLP features
# and finally generate a vrt file with the name of the audio/video file + .vrt
def generate_vrt(file, whisper_model):

  print('Loading whisper model...')
  model = whisper.load_model(whisper_model)

  print('Transcribing audio...')
  results = model.transcribe(file, max_initial_timestamp=None)

  aligned_results = align_audio_text(file,results['text'])
  word_timestamps = get_word_timestamps(aligned_results) 

  print('Loading spacy model...')
  nlp = spacy.load('en_core_web_lg')
  doc = nlp(results['text'])
  sentencizer = nlp.add_pipe("sentencizer")

  vrt_file = open(file + '.vrt', 'w')
  # file without extension and path
  filename = file.split('/')[-1].split('.')[0]
  vrt_file.write('<text id="' + filename  +  '" '  + 'file="' + file + '" '  + ' language="' + results['language'] + '">\n')

  print('Writing to vrt file...')
  last_start = 0
  last_end = 0
  sentence_id = 0
  for sent in doc.sents:
      sentence_id += 1
      vrt_file.write('<s id="' + str(sentence_id) + '" ' + 'file="' + filename + '" ' + ">\n")
      for token in sent:
        # find the word timestamp for the token
        found = False
        for word in word_timestamps:
          # if word starts with token print token and timestamp
          if word['word'].startswith(token.text):
            found = True
            last_start = word['start']
            last_end = word['end']
            vrt_file.write(token.text + " \t " + token.lower_  +  " \t " + token.prefix_  +  " \t " + token.suffix_ + " \t " + 
                          str(token.is_digit) + " \t " + str(token.like_num) + " \t " + token.dep_ + " \t " + token.shape_ + " \t " + 
                          token.lemma_ + " \t " +  token.pos_ + " \t " +  token.tag_ + " \t "  +  str(token.sentiment) + " \t " +
                          str(token.is_alpha) + " \t " +  str(token.is_stop) + " \t " +  token.head.text + " \t " +  
                          token.head.pos_ + " \t " +  str([child for child in token.children]) + " \t " + 
                          get_secs(word['start']) + " \t " + get_msecs(word['start']) + " \t " + get_secs(word['end']) + " \t " + get_msecs(word['end']) + "\n")
            # delete all words until the used one
            word_timestamps = word_timestamps[word_timestamps.index(word)+1:]
            break
        if not found:
            vrt_file.write(token.text + " \t " + token.lower_  +  " \t " + token.prefix_  +  " \t " + token.suffix_ + " \t " + 
                          str(token.is_digit) + " \t " + str(token.like_num) + " \t " + token.dep_ + " \t " + token.shape_ + " \t " + 
                          token.lemma_ + " \t " +  token.pos_ + " \t " +  token.tag_ + " \t "  +  str(token.sentiment) + " \t " +
                          str(token.is_alpha) + " \t " +  str(token.is_stop) + " \t " +  token.head.text + " \t " +  
                          token.head.pos_ + " \t " +  str([child for child in token.children]) + " \t " + 
                          get_secs(last_start) + " \t " + get_msecs(last_start) + " \t " + get_secs(last_end) + " \t " + get_msecs(last_end) + "\n")
      vrt_file.write("</s>\n")
  vrt_file.write("</text>\n")
  vrt_file.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--file', type=str, help='audio or video file')
  parser.add_argument('--whisper_model', type=str, help='whisper model', default='large')
  args = parser.parse_args()

  generate_vrt(args.file, args.whisper_model)

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
from thefuzz import fuzz
import re 

# Get and audio/video file and a transcription. Align the audio/video with the transcription
# and return gentle's result
def align_audio_text(audio_file, transcription):

  disfluencies = set(['uh', 'um'])
  resources = gentle.Resources()
  with gentle.resampled(audio_file) as wavfile:
      print("starting alignment")
      aligner = gentle.ForcedAligner(resources, transcription, nthreads=4, disfluency=False, conservative=True, disfluencies=False)
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

# Load spacy model and return the model of the language
def load_spacy_model(language):
  if language == 'es':
    print("Loading spacy model for Spanish")
    return spacy.load('es_core_news_lg')
  if language == 'de':
    print("Loading spacy model for German")
    return spacy.load('de_core_news_lg')
  if language == 'pl':
    print("Loading spacy model for Polish")
    return spacy.load('pl_core_news_lg')
  else:
    print("Loading spacy model for English")
    return spacy.load('en_core_web_lg')

# Parse data from file name expecting a filename as this: 2016-01-01_0000_US_MSNBC_Hardball_with_Chris_Matthews.mp4
def parse_file(file):
  # file without extension and path
  filename = file.split('/')[-1].split('.')[0]
  # sub - for _ in filename
  filename = filename.replace('-', '_')  
  filename_with_ext = file.split('/')[-1]
  # filename with extension: 2016-01-01_0000_US_MSNBC_Hardball_with_Chris_Matthews.mp4
  # date and time: 2016-01-01_0000
  date_time = filename_with_ext.split('_')[0] + '_' + filename_with_ext.split('_')[1]
  # channel: MSNBC
  channel = filename_with_ext.split('_')[3]
  # title: Hardball_with_Chris_Matthews from 2016-01-01_0000_US_MSNBC_Hardball_with_Chris_Matthews.mp4
  title = '_'.join(filename_with_ext.split('_')[4:]).split('.')[0]  
  # year: 2016
  year = date_time.split('-')[0]
  # month: 01
  month = date_time.split('-')[1]
  # day: 01
  day = date_time.split('-')[2].split('_')[0]
  # time: 0000
  time = date_time.split('-')[2].split('_')[1]
  return filename, filename_with_ext, date_time, channel, title, year, month, day, time

# Get an audio/video file and generate a transcription using whisper
# then align the audio with the transcription using gentle.
# Process transcription with spacy to get NLP features
# and finally generate a vrt file with the name of the audio/video file + .vrt
def generate_vrt(file, whisper_model):
  print('Loading whisper model...')
  model = whisper.load_model(whisper_model)

  print('Transcribing audio...')
  results = model.transcribe(file, max_initial_timestamp=None)

  # Load spacy model
  nlp = load_spacy_model(results['language'])
  nlp.add_pipe("sentencizer")
  doc = nlp(results['text'])

  # we use spacy tokens to pass it to gentle
  # gentle mess spacy tokenization with its own tokenization
  # so we remove puntuation from not punctuation tokens
  spacy_tokens = ""
  for token in doc:
    # if token isn't punctuation delete , - or . 
    if token.pos_ != 'PUNCT':
      # delete puntuation from token
      tmp = token.text.replace(',', '')
      tmp = tmp.replace('.', '')
      tmp = tmp.replace('-', '')
      spacy_tokens += tmp + " "
    else:
      spacy_tokens += token.text + " "

  aligned_results = align_audio_text(file,spacy_tokens) 
  words_timestamps = get_word_timestamps(aligned_results)

  # write transcription to a file to debug if needed
  with open(file + '.txt', 'w') as f:
    f.write(spacy_tokens)
  f.close()
  # write word timestamps to a file to debug if needed
  with open(file + '.timestamps', 'w') as f:
    for word in words_timestamps:
      f.write(word['word'] + ' ' + word['start'] + ' ' + word['end'] + '\n')
  f.close()

  print('Writing to vrt file...')
  vrt_file = open(file + '.vrt', 'w')
  filename, filename_with_ext, date_time, channel, title, year, month, day, time = parse_file(file)
  vrt_file.write('<text id="' + filename  +  '" '  + 'file="' + filename_with_ext + '" '  + ' language="' + results['language'] + '" ' +
      'collection="Daedalus Test" ' + 'date="' + date_time + '" ' + 'channel="' + channel + '" ' + 'title="' + title + '" ' + 'year="' + year + '" ' + 'month="' + month + '" ' + 'day="' + day + '" ' + 'time="' + time + '" ' + '>\n')  
  vrt_file.write('<story>\n')
  vrt_file.write('<turn>\n')
  last_start = 0
  last_end = 0
  last_token = ''
  sentence_id = 0
  for sent in doc.sents:
      sentence_id += 1
      vrt_file.write('<s id="' + str(sentence_id) + '" ' + 'file="' + filename + '" ' +     
      'reltime=' + get_secs(last_start) + '" '  ">\n")
      for token in sent:
        if token.pos_ != 'PUNCT':
          # regex remove everythink but letters and numbers and ' from token
          token_text = re.sub('[^A-Za-záéíóúÁÉÍÓÚüÜ0-9\']+', '', token.text)      
        else:
          token_text = token.text          
        # find the word timestamp for the token
        found = False
        for word in words_timestamps:
          # Use thefuzz library to find if word and token are the "same"
          if fuzz.ratio(word['word'], token_text) > 60:
            found = True
            last_start = word['start']
            last_end = word['end']
            last_token = token_text
            vrt_file.write(token_text + " \t " + token.lower_  +  " \t " + token.prefix_  +  " \t " + token.suffix_ + " \t " + 
                          str(token.is_digit) + " \t " + str(token.like_num) + " \t " + token.dep_ + " \t " + token.shape_ + " \t " + 
                          token.lemma_ + " \t " +  token.pos_ + " \t " +  token.tag_ + " \t "  +  str(token.sentiment) + " \t " +
                          str(token.is_alpha) + " \t " +  str(token.is_stop) + " \t " +  token.head.text + " \t " +  
                          token.head.pos_ + " \t " +  str([child for child in token.children]) + " \t " + 
                          get_secs(word['start']) + " \t " + get_msecs(word['start']) + " \t " + get_secs(word['end']) + " \t " + get_msecs(word['end']) + "\n")
              # delete all words until the used one
            words_timestamps = words_timestamps[words_timestamps.index(word)+1:]
            break
          if not found:
            if fuzz.ratio(last_token, word['word']) > 20:
              words_timestamps = words_timestamps[words_timestamps.index(word)+2:]
            vrt_file.write(token_text + " \t " + token.lower_  +  " \t " + token.prefix_  +  " \t " + token.suffix_ + " \t " + 
                            str(token.is_digit) + " \t " + str(token.like_num) + " \t " + token.dep_ + " \t " + token.shape_ + " \t " + 
                            token.lemma_ + " \t " +  token.pos_ + " \t " +  token.tag_ + " \t "  +  str(token.sentiment) + " \t " +
                            str(token.is_alpha) + " \t " +  str(token.is_stop) + " \t " +  token.head.text + " \t " +  
                            token.head.pos_ + " \t " +  str([child for child in token.children]) + " \t " + 
                            get_secs(last_start) + " \t " + get_msecs(last_start) + " \t " + get_secs(last_end) + " \t " + get_msecs(last_end) + "\n")
            break
      vrt_file.write("</s>\n")
  vrt_file.write('</turn>\n')
  vrt_file.write('</story>\n')
  vrt_file.write("</text>\n")
  vrt_file.close()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--file', type=str, help='audio or video file')
  parser.add_argument('--whisper_model', type=str, help='whisper model', default='large')
  args = parser.parse_args()

  generate_vrt(args.file, args.whisper_model)

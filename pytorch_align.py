#!/usr/bin/env python3

import torch
import torchaudio
from datetime import timedelta
from dataclasses import dataclass
from srt import Subtitle, compose
import whisper
from pydub import AudioSegment
import re
import num2words
import argparse
import random
import string
import os
import unidecode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

torch.random.manual_seed(0)

def force_align(SPEECH_FILE, transcript, start_index, start_time):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    with torch.inference_mode():
        waveform, _ = torchaudio.load(SPEECH_FILE)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]


    def get_trellis(emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1))
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")

        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis


    trellis = get_trellis(emission, tokens)

    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float


    def backtrack(trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]


    path = backtrack(trellis, emission, tokens)

    # Merge the labels
    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        @property
        def length(self):
            return self.end - self.start


    def merge_repeats(path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments


    segments = merge_repeats(path)

    # Merge words
    def merge_words(segments, separator="|"):
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words

    word_segments = merge_words(segments)
    subs = []
    for i,word in enumerate(word_segments):
        ratio = waveform.size(1) / (trellis.size(0) - 1)
        x0 = int(ratio * word.start)
        x1 = int(ratio * word.end)
        start = timedelta(seconds=start_time + x0 / bundle.sample_rate)
        end = timedelta(seconds=start_time + x1 / bundle.sample_rate )
        #subtitle = Subtitle(start_index+i, start, end, word.label)
        subs.append({"word":word.label.lower(), "start":str(round(start.total_seconds(),2)), "end":str(round(end.total_seconds(),2))})
        # print(f" ({start} --> {end}) : " + word.label)

    #print(compose(subs))
    return subs


# Function gets results from whisper output and mp4 file
def pytorch_force_align(results, mp4_file):
    # create temp wav file from mp4
    tmp_filename = '/tmp/' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".wav"
    print("TMP FILENAME: " + tmp_filename)
    ffmpeg_command = "ffmpeg -i " + mp4_file + " -ab 16k -ac 2 -ar 44100 -vn " + tmp_filename + " -y"
    os.system(ffmpeg_command)

    segments = results["segments"]

    start_index = 0
    total_subs = []
    for i,segment in enumerate(segments):
        text = segment["text"]
        audioSegment = AudioSegment.from_wav(tmp_filename)[segment["start"]*1000:segment["end"]*1000]
        audioSegment.export(str(i)+'.wav', format="wav") #Exports to a wav file in the current path.
        # Clean transcription 
        transcript=text.strip().replace(" ", "|")
        transcript = re.sub(r'[^\w|\s]', '', transcript)
        transcript = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), transcript)
        transcript = re.sub(r'\s+', '|', transcript)
        transcript = re.sub(r',', '', transcript)
        # change all accents to ascii
        transcript = unidecode.unidecode(transcript)

        subs = force_align(str(i)+'.wav', transcript.upper(), start_index, segment["start"])
        start_index += len(segment["text"])
        # delete str(i)+'.wav' file
        os.unlink(str(i)+'.wav')
        total_subs += subs
    # delete tmp wav file
    os.unlink(tmp_filename)
    return total_subs


if __name__ == "__main__":

    # Require file input with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to audio file")

    args = parser.parse_args()

    mp4_file = args.file           


    model = whisper.load_model("large")
    transcription = model.transcribe(mp4_file)
    # ffmpeg mp4_file to wav file
    # create random string for temp wav file
    tmp_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    ffmpeg_command = "ffmpeg -i " + mp4_file + " -ab 16k -ac 2 -ar 44100 -vn " + tmp_filename + ".wav -y 2> /dev/null "
    os.system(ffmpeg_command)

    # print the recognized text
    segments = transcription["segments"]
    # print("Transcription complete:")
    # print(transcription["text"])
    # print("Starting to force alignment...")
    start_index = 0
    total_subs = []
    for i,segment in enumerate(segments):
        text = segment["text"]
        audioSegment = AudioSegment.from_wav(tmp_filename + ".wav")[segment["start"]*1000:segment["end"]*1000]
        audioSegment.export(str(i)+'.wav', format="wav") #Exports to a wav file in the current path.
        transcript=text.strip().replace(" ", "|")
        # replace , with space in transcript
        #transcript = transcript.replace(",", " ")
        transcript = re.sub(r'[^\w|\s]', '', transcript)
        transcript = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), transcript)
        transcript = re.sub(r'\s+', '|', transcript)
        transcript = re.sub(r',', '', transcript)
        # change all accents to ascii
        transcript = unidecode.unidecode(transcript)

        
        # print("WAV FILE: " + str(i) + ".wav")
        print(f'{segment["start"]:.2f}' + " : " + text)
        # print(segment)

        
        subs = force_align(str(i)+'.wav', transcript.upper(), start_index, segment["start"])
        start_index += len(segment["text"])
        # delete str(i)+'.wav' file
        os.unlink(str(i)+'.wav')
        total_subs += subs



        #total_subs.extend(subs)
    # out of for loop
    # delete tmp_filename.wav file
    os.unlink(tmp_filename + ".wav")
    
    # write total_subs in file
    with open("subs.timestamps", "w") as f:
        for sub in total_subs:
            f.write(sub["word"] + " " + sub["start"] + " " + sub["end"] + "\n")



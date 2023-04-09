# Parts of the code are adapted from the snippets provided in the TorchAudio Wav2Vec forced alignment tutorial.
# The full tutorial can be found here: https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html

import argparse
import os
from dataclasses import dataclass

import torch
import torchaudio
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCTC, AutoProcessor


class Wav2Vec2Aligner:
    def __init__(self, model_name, input_wavs_sr, cuda):
        self.cuda = cuda
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()
        if self.cuda:
            self.model.to(device="cuda")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.resampler = torchaudio.transforms.Resample(input_wavs_sr, 16_000)
        blank_id = 0
        vocab = list(self.processor.tokenizer.get_vocab().keys())
        for i in range(len(vocab)):
            if vocab[i] == "[PAD]" or vocab[i] == "<pad>":
                blank_id = i
        print("Blank Token id [PAD]/<pad>", blank_id)
        self.blank_id = blank_id

    def speech_file_to_array_fn(self, wav_path):
        speech_array, sampling_rate = torchaudio.load(wav_path)
        speech = self.resampler(speech_array).squeeze().numpy()
        return speech

    def align_single_sample(self, item):
        blank_id = self.blank_id
        transcript = "|".join(item["sent"].split(" "))
        if not os.path.isfile(item["wav_path"]):
            print(item["wav_path"], "not found in wavs directory")

        speech_array = self.speech_file_to_array_fn(item["wav_path"])
        inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        if self.cuda:
            inputs = inputs.to(device="cuda")

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()

        # get labels from vocab
        labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
            :-1
        ]  # logits don't align with the tokenizer's vocab

        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])

        def get_trellis(emission, tokens, blank_id=0):
            """
            Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
            that represents the probabilities of each source token being at a certain time step
            """
            num_frames = emission.size(0)
            num_tokens = len(tokens)

            # Trellis has extra diemsions for both time axis and tokens.
            # The extra dim for tokens represents <SoS> (start-of-sentence)
            # The extra dim for time axis is for simplification of the code.
            trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
            trellis[:, 0] = 0
            for t in range(num_frames):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            return trellis

        trellis = get_trellis(emission, tokens, blank_id)

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, tokens, blank_id=0):
            """
            Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path
            """
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

        path = backtrack(trellis, emission, tokens, blank_id)

        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return f"{self.label}\t{self.score:4.2f}\t{self.start*20:5d}\t{self.end*20:5d}"

            @property
            def length(self):
                return self.end - self.start

        def merge_repeats(path):
            """
            Merge repeated tokens into a single segment. Note: this shouldn't affect repeated characters from the
            original sentences (e.g. `ll` in `hello`)
            """
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
        #the_result = [('we','start_sample','end_sample'),('word','start_sample','end_sample')]
        the_result = []
        word = ''
        first_char=True
        for seg in segments:
            parts = str(seg).split('\t')
            character = parts[0]
            start_ms = parts[2]
            end_ms = parts[3]
            start_point = int(start_ms)*0.001*16000
            end_point = int(end_ms)*0.001*16000
            #start_point = int(start_ms)*0.001 #show seconds format
            #end_point = int(end_ms)*0.001 #show seconds format
            if character !='|':
                word = word+character
                if first_char:
                    w_s = start_point 
                    first_char=False
                #w_e = end_point
            else: # reset
                w_e = end_point
                the_result.append((word, int(w_s), int(w_e)))
                #the_result.append((word, w_s, w_e))
                word = ''
                w_s = 0
                w_e = 0
                first_char = True
        the_result.append((word, int(w_s), int(w_e)))
        #the_result.append((word, w_s, w_e))
        return the_result

    def align_data(self, wav_dir, text_file, output_dir):

        assert len(wav_dir)==len(text_file)

        items = []
        for wav_idx in range(len(wav_dir)):
            wav_path = wav_dir[wav_idx]
            sentence = text_file[wav_idx]
            wav_name = os.path.basename(wav_path).replace('.wav','.txt')
            out_path = os.path.join(output_dir, wav_name)
            items.append({"sent": sentence, "wav_path": wav_path, "out_path": out_path})
        print("Number of samples", len(items))

        result = []
        for item in tqdm(items):
            the_result = self.align_single_sample(item)
            result.append(the_result)

        return result      


def main():

    model_name = 'arijitx/wav2vec2-xls-r-300m-bengali'
    input_wavs_sr = 16000
    cuda = False
    aligner = Wav2Vec2Aligner(model_name, input_wavs_sr, cuda)

    wav_path = ['path/to/wav1.wav','path/to/wav2.wav']
    transcript_list = ['transcript of wav1', 'transcript of wav2']
    result = aligner.align_data(wav_path, transcript_list, '')
    print(result) 
    #result:[ 
    # [(transcript, start_sample_point, end_sample_point), (of, start_sample_point, end_sample_point), (wav1, start_sample_point, end_sample_point)],
    # [(transcript, start_sample_point, end_sample_point), (of, start_sample_point, end_sample_point), (wav2, start_sample_point, end_sample_point)]
    #]
    


if __name__ == "__main__":
    main()

# Copyright 2023 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Format Treebank3 annotation (.dff) to tsv
"""
import argparse
import os
import pandas as pd


def remove_comments(tokens):
    """Remove `<<`, `>>` and its contents from `tokens`.
    """
    tokens_ = []
    level = 0
    for token in tokens:
        if "<<" in token:
            level += 1
            assert level == 1
        if ">>" in token:
            level -= 1
        if level == 0:
            tokens_.append(token)
    assert level == 0
    return tokens_


def parse_tokens(tokens):
    """Parse `tokens` to generate
    tokens without any tags (`tokens_`) and their disfluency labels (`labels_`).
    """
    tokens = [token for token in tokens if token not in ["/", "#", "--", "-/"]]
    tokens = [token for token in tokens if not token.startswith("<")]  # <laughter>
    tokens = [token for token in tokens if not token.endswith(">")]

    level = 0
    tokens_ = []
    labels_ = []

    # Non-sentence elements (e.g. {F Uh, } first, )
    # See https://www.cs.brandeis.edu/~cs140b/CS140b_docs/DysfluencyGuide.pdf
    for token in tokens:
        if token in ["{F", "{E", "{D", "{C", "{A"]:
            level += 1
        elif token == "}":
            level -= 1
        elif level >= 1:
            tokens_.append(token)
            labels_.append(1)
        else:
            tokens_.append(token)
            labels_.append(0)
    assert level == 0

    tokens = tokens_
    labels = labels_

    # Restarts (e.g. how do you feel [ about, + {F uh, } about ] )
    level = 0
    MAX_LEVEL = 100
    beforeIPlv = [True] * MAX_LEVEL
    tokens_, labels_ = [], []
    for token, label in zip(tokens, labels):
        if token == "[":
            level += 1
            assert level <= MAX_LEVEL
            beforeIPlv[level] = True
        elif token == "+":
            beforeIPlv[level] = False
        elif token == "]":
            level -= 1
        elif level >= 1 and beforeIPlv[level]:
            tokens_.append(token)
            labels_.append(1)
        else:
            tokens_.append(token)
            labels_.append(label)
    assert level == 0
    tokens, labels = tokens_, labels_

    tokens = [token.lower() for token in tokens]

    for i in range(len(tokens)):
        token = tokens[i]
        token_ = ""
        for char in token:
            if char.isalpha() or char.isdigit() or char in ["&", "'", "-", "/"]:
                token_ += char
        tokens[i] = token_

    tokens_, labels_ = [], []
    for token, label in zip(tokens, labels):
        if token:
            tokens_.append(token)
            labels_.append(label)
    tokens, labels = tokens_, labels_

    return tokens, labels


def get_dysfl_label(dff_file_path):
    """Get tokens and disfluency labels
    for each file (`dff_file_path`) that contains two speakers (A/B).
    """
    with open(dff_file_path) as f:
        lines = [line.strip() for line in f]

    speaker = ""
    tokens_a = []
    tokens_b = []

    for line in lines[31:]:  # Skip metadata at the beginning of .dff files
        if line:
            if len(line.split(": ")) >= 2 and line[0] in ["A", "B", "@"]:
                speaker_num = line.split(": ")[0]  # A.1:
                # Some contains `: ` expression in sentence (e.g. <<sounds like: t, t, t, t, t>>)
                sent = " ".join(line.split(": ")[1:])
                sent = sent.strip()
                if len(speaker_num.split(".")) > 1:
                    speaker, _ = tuple(speaker_num.split("."))
                else:
                    # In some cases, numbers are missing (e.g. A: )
                    speaker = speaker_num
                speaker = speaker.replace("@", "")
                assert speaker in ["A", "B"]
            else:
                sent = line

            tokens = sent.split()

            if speaker == "A":
                tokens_a.extend(tokens)
            elif speaker == "B":
                tokens_b.extend(tokens)

    # Remove << >> (e.g. <<sounds like: t, t, t, t, t>>)
    tokens_a = remove_comments(tokens_a)
    tokens_b = remove_comments(tokens_b)

    tokens_a, labels_a = parse_tokens(tokens_a)
    assert len(tokens_a) == len(labels_a)
    tokens_b, labels_b = parse_tokens(tokens_b)
    assert len(tokens_b) == len(labels_b)

    return tokens_a, tokens_b, labels_a, labels_b


def main(args):
    results = []  # speaker_id, text, label

    for dir in os.listdir(args.dff_dir):
        for file in os.listdir(os.path.join(args.dff_dir, dir)):
            tokens_a, tokens_b, labels_a, labels_b = get_dysfl_label(
                os.path.join(os.path.join(args.dff_dir, dir), file)
            )

            session_id = file.replace(".dff", "")[2:]
            session_id = f"sw0{session_id}"
            speaker_id_a = f"{session_id}-A"
            results.append(
                (speaker_id_a, " ".join(tokens_a), " ".join(list(map(str, labels_a))))
            )
            speaker_id_b = f"{session_id}-B"
            results.append(
                (speaker_id_b, " ".join(tokens_b), " ".join(list(map(str, labels_b))))
            )

    df = pd.DataFrame(results, columns=["speaker_id", "text", "isdysfl"])
    df.to_csv(args.output_tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dff_dir", type=str)
    parser.add_argument("output_tsv_path", type=str)
    args = parser.parse_args()
    main(args)

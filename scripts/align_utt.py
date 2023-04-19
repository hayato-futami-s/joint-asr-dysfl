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

""" Align disfluency labels (tsv) to ESPnet utterances with BERT re-tokenization
"""
import argparse
import copy
import json
import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def get_alignment(text1, text2):
    """Get alignment (and number of errors) between `text1` and `text2`
    by backtracking the path of the smallest edit distance.
    """
    # Edit distance
    d = np.zeros((len(text2) + 1) * (len(text1) + 1), dtype=np.uint16)
    d = d.reshape((len(text2) + 1, len(text1) + 1))

    for i in range(len(text2) + 1):
        for j in range(len(text1) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(text2) + 1):
        for j in range(1, len(text1) + 1):
            if text2[i - 1] == text1[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)
    dist = d[len(text2)][len(text1)]

    # Backtrack
    x = len(text2)
    y = len(text1)
    alignment = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and text2[x - 1] == text1[y - 1]:
                    alignment.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    alignment.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    alignment.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    alignment.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    alignment.append("I")
                    y = y - 1
                else:
                    alignment.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                alignment.append("D")
                x = x - 1
            else:
                raise ValueError
    alignment.reverse()

    n_sub = alignment.count("S")
    n_ins = alignment.count("I")
    n_del = alignment.count("D")
    n_cor = alignment.count("C")
    assert dist == (n_sub + n_ins + n_del)

    return alignment, n_sub, n_ins, n_del, n_cor


def split_utterances(text1, text2, labels2, utt_info, alignment):
    """Split speaker-level text (`text1` and `text2`) and labels (`labels2`)
    into utterance-level ones based on `utt_info` and `alignment`.
    """
    idx1 = 0  # Index for text1
    idx2 = 0  # Index for text2 (and labels2)
    idx_utt = -1
    ntokens_utt = 0
    text1_utt, text2_utt, labels2_utt = [], [], []
    utt_id = None

    utts = {}

    for alitype in alignment:
        # Move to next utterance
        if ntokens_utt - len(text1_utt) == 0:
            if utt_id is not None:
                utts[utt_id] = (text1_utt, text2_utt, labels2_utt)
            idx_utt += 1
            if idx_utt >= len(utt_info):
                break
            utt_id = utt_info[idx_utt][0]
            ntokens_utt = utt_info[idx_utt][1]
            text1_utt = []
            text2_utt = []
            labels2_utt = []

        if alitype == "D":  # Deletion
            idx2 += 1
        elif alitype == "I":  # Insertion
            text1_utt.append(text1[idx1])
            idx1 += 1
        else:
            text1_utt.append(text1[idx1])
            idx1 += 1
            text2_utt.append(text2[idx2])
            labels2_utt.append(labels2[idx2])
            idx2 += 1
    if utt_id is not None:
        utts[utt_id] = (text1_utt, text2_utt, labels2_utt)

    return utts


def main(args):
    dysfl_tsv = pd.read_table(args.dysfl_tsv)

    espnet_data_paths = args.espnet_data_paths.split(",")

    # The script accepts multiple json files split by comma.
    # Concat all the utterances in these json files to make one json file.
    data_utt_json = None
    for espnet_data_path in espnet_data_paths:
        assert espnet_data_path.endswith(".json")
        with open(espnet_data_path, "rb") as f:
            data_utt_json_ = json.load(f)

        if data_utt_json is None:
            data_utt_json = data_utt_json_
        else:
            for utt_id, utt_json in data_utt_json_["utts"].items():
                data_utt_json["utts"][utt_id] = utt_json

    uttbyspk = defaultdict(
        list
    )  # speaker_id -> its json, to make `textbyspk`, `uttinfobyspk`

    for utt_id, utt_json in data_utt_json["utts"].items():
        # This script do not support speed perturb data.
        # Apply speed perturbation afterwards.
        assert re.match(r"sp\d+.\d+-", utt_id) is None

        speaker_id = utt_id.split("_")[0]
        uttbyspk[speaker_id].append((utt_id, utt_json))

    textbyspk = defaultdict(list)  # speaker_id -> text
    uttinfobyspk = defaultdict(
        list
    )  # speaker_id -> list of (utt_id, the number of tokens)
    for speaker_id, speaker_utts in uttbyspk.items():
        # Sort by starting time in utt_id
        speaker_utts.sort(key=lambda x: int(x[0].split("_")[1].split("-")[0]))
        for utt_id, utt_json in speaker_utts:
            text = utt_json["output"][0]["text"].split()
            # Remove tags such as [noise] and [laughter]
            text = [token for token in text if not token.startswith("[")]
            if len(text) > 0:
                textbyspk[speaker_id].extend(text)
                uttinfobyspk[speaker_id].append((utt_id, len(text)))

    aligned_utts = {}  # utt_id -> (text_orig, text_dysfl, labels_dysfl)

    for i, row in enumerate(dysfl_tsv.itertuples()):  # Loop for spakers
        if row.speaker_id in textbyspk:
            print(f"({(i+1):d}/{len(dysfl_tsv):d}) {row.speaker_id}", flush=True)

            text_orig = textbyspk[row.speaker_id]
            utt_info = uttinfobyspk[row.speaker_id]
            text_dysfl = row.text.split()
            labels_dysfl = row.isdysfl.split()
            assert len(text_dysfl) == len(labels_dysfl)

            alignment, n_sub, n_ins, n_del, n_cor = get_alignment(text_orig, text_dysfl)
            print(f"Edit distance: S={n_sub:d} I={n_ins:d} D={n_del:d} C={n_cor:d}")

            # Threshold to reject speaker due to too many mismatch between two texts
            REJECT_THRE = 4
            if n_cor < (n_sub + n_ins + n_del) * REJECT_THRE:
                # Note that speaker_id (A/B) swapping can exist
                # So re-try alignment for swapped speaker_id
                if row.speaker_id.endswith("A"):
                    speaker_id = row.speaker_id.replace("A", "B")
                else:
                    speaker_id = row.speaker_id.replace("B", "A")
                print(f"! Swap speaker {row.speaker_id} <-> {speaker_id}")

                text_orig = textbyspk[speaker_id]
                utt_info = uttinfobyspk[speaker_id]
                text_dysfl = row.text.split()
                labels_dysfl = row.isdysfl.split()
                assert len(text_dysfl) == len(labels_dysfl)

                # Alignment between `text_orig` (from ESPnet) and `text_dysfl` (from Treebank3)
                alignment, n_sub, n_ins, n_del, n_cor = get_alignment(
                    text_orig, text_dysfl
                )
                print(f"Edit distance: S={n_sub:d} I={n_ins:d} D={n_del:d} C={n_cor:d}")

                if n_cor < (n_sub + n_ins + n_del) * REJECT_THRE:
                    print(
                        f"! Skip speaker {row.speaker_id}/{speaker_id} (there are too many errors)"
                    )
                    continue

            # speaker -> utterances
            aligned_utts.update(
                split_utterances(
                    text_orig, text_dysfl, labels_dysfl, utt_info, alignment
                )
            )

        else:
            print(f"! Missing {row.speaker_id}")

    ali_data_utt_json = copy.deepcopy(data_utt_json)

    for utt_id, utt_json in data_utt_json["utts"].items():
        if utt_id in aligned_utts:
            _, text_ali, labels_ali_ = aligned_utts[utt_id]

            if not text_ali:
                ali_data_utt_json["utts"].pop(utt_id)
                continue

            # Remove partial word specifier in advance
            text_ali = [
                token[:-1] if token.endswith("-") else token for token in text_ali
            ]

            ali_data_utt_json["utts"][utt_id]["output"][0]["text"] = " ".join(text_ali)

            assert len(text_ali) == len(labels_ali_)

            # Re-tokenize with BertTokenizer
            token_ali, tokenid_ali = [], []
            labels_ali = []
            for token_orig, label_orig in zip(text_ali, labels_ali_):
                token = tokenizer.tokenize(token_orig)
                tokenid = tokenizer.convert_tokens_to_ids(token)
                token_ali.extend(token)
                tokenid_ali.extend(tokenid)
                labels_ali.extend([label_orig] * len(token))

            ali_data_utt_json["utts"][utt_id]["output"][0]["token"] = " ".join(
                token_ali
            )
            ali_data_utt_json["utts"][utt_id]["output"][0]["tokenid"] = " ".join(
                list(map(str, tokenid_ali))
            )
            ali_data_utt_json["utts"][utt_id]["output"][0]["shape"] = [
                len(token_ali),
                len(tokenizer),
            ]

            # Disfluency annotation
            ali_data_utt_json["utts"][utt_id]["output"][0]["isdysfl"] = " ".join(
                labels_ali
            )
            assert len(token_ali) == len(tokenid_ali) == len(labels_ali)
        else:
            ali_data_utt_json["utts"].pop(utt_id)

    output_dir = os.path.dirname(espnet_data_paths[0])
    output_path = os.path.join(output_dir, "data_all_dysfl.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ali_data_utt_json, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dysfl_tsv", type=str)
    parser.add_argument("espnet_data_paths", type=str)
    args = parser.parse_args()
    main(args)

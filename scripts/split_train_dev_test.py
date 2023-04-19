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

""" Split train/dev/test data following https://aclanthology.org/2020.acl-main.346/
"""
import argparse
import json
import os


def main(args):
    assert args.data_path.endswith(".json")
    with open(args.data_path, "rb") as f:
        data_all = json.load(f)

    data_train = {"utts": {}}
    data_dev = {"utts": {}}
    data_test = {"utts": {}}

    for utt_id, utt_json in data_all["utts"].items():
        # Test
        if utt_id.startswith("sw040") or utt_id.startswith("sw041"):
            data_test["utts"][utt_id] = utt_json
        # Dev
        elif (
            utt_id.startswith("sw045")
            or utt_id.startswith("sw046")
            or utt_id.startswith("sw047")
            or utt_id.startswith("sw048")
            or utt_id.startswith("sw049")
        ):
            data_dev["utts"][utt_id] = utt_json
        # Train
        elif utt_id.startswith("sw02") or utt_id.startswith("sw03"):
            data_train["utts"][utt_id] = utt_json

    output_dir = os.path.dirname(args.data_path)
    data_train_path = os.path.join(output_dir, "data_train_dysfl.json")
    data_dev_path = os.path.join(output_dir, "data_dev_dysfl.json")
    data_test_path = os.path.join(output_dir, "data_test_dysfl.json")

    with open(data_train_path, "w", encoding="utf-8") as f:
        json.dump(data_train, f, ensure_ascii=False, indent=4)
    with open(data_dev_path, "w", encoding="utf-8") as f:
        json.dump(data_dev, f, ensure_ascii=False, indent=4)
    with open(data_test_path, "w", encoding="utf-8") as f:
        json.dump(data_test, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()
    main(args)

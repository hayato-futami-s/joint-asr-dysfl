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

espnet_data=$1
treebank3_dir=$2

if [ $# != 2 ]; then
  echo "Usage:"
  echo "  $0 <espnet_data> <treebank3_dir>"
  echo "e.g.:"
  echo "  $0 dump/train_nodup/deltafalse/data_bpe2000.json,dump/train_dev_trim/deltafalse/data_bpe2000.json data/treebank_3"
  exit 1;
fi

espnet_data0=(${espnet_data//,/ })
output_dir="$(dirname ${espnet_data0})"

treebank3_dff="${treebank3_dir}/dysfl/dff/swbd"
dysfl_tsv_path="${output_dir}/treebank3_dysfl_label.tsv"

data_all_dysfl="${output_dir}/data_all_dysfl.json"

# Format Treebank3 annotation (.dff) to tsv
python scripts/format_dysfl_label.py ${treebank3_dff} ${dysfl_tsv_path}

# Align disfluency labels (tsv) to ESPnet utterances with BERT re-tokenization
python scripts/align_utt.py ${dysfl_tsv_path} ${espnet_data}

# Split train/dev/test
python scripts/split_train_dev_test.py ${data_all_dysfl}

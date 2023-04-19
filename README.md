This is a script for disfluency-annotated paired speech and text data on Switchboard corpus.

The script adds Treebank3 disfluency annotation to ESPnet-style transcripts.

See https://arxiv.org/abs/2211.08726

## Usage

```
bash scripts/run.sh ${espnet_data} ${treebank3_dir}
```

As `${espnet_data}`, specify JSON files in ESPnet Switchboard recipe `egs/swbd/asr1/`.
(e.g. `dump/train_nodup/deltafalse/data_bpe2000.json`, `dump/train_dev_trim/deltafalse/data_bpe2000.json`)
Specify both train and dev files split by `,`.

JSON data (ESPnet) format should be:
```
{
    "utts": {
        "sw0xxxx-A_xxxxxx-xxxxxx": {
            "input": [
                ...
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        ...
                    ],
                    "text": " ... ",
                    "token": " ... ",
                    "tokenid": " ... "
                }
            ],
            "utt2spk": ...
        },
        ...
    }
}
```

As `${treebank3_dir}`, specify [Treebank3 corpus](https://catalog.ldc.upenn.edu/LDC99T42) directory.
The script reads disfluency annotation from `.dff` files at `dysfl/dff/swbd/{2/3/4}/`.
Its format should be like: https://catalog.ldc.upenn.edu/desc/addenda/LDC99T42.dff.txt

Generated data will be output to the same directory as `${espnet_data}` as `data_{train/dev/test}_dysfl.json`.
Disfluency annotation for each token (`isdysfl`) will be added to JSON.

Note that disfluency annotation is not always correct because of parse or alignment errors.

## Requirements

Python 3.9.13, pip 22.2.2
```
pip install -r requirements.txt
```

## Licence

This repository is provided under the [Apache License Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

## Citation

```
@article{Futami2022StreamingJS,
  title={Streaming Joint Speech Recognition and Disfluency Detection},
  author={Hayato Futami and Emiru Tsunoo and Kentaro Shibata and Yosuke Kashiwagi and Takao Okuda and Siddhant Arora and Shinji Watanabe},
  journal={ArXiv},
  year={2022},
}
```

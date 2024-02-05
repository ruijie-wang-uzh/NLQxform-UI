#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/30 19:51
# @Author  : Zhiruo Zhang
# @File    : heldout_test.py
# @Description : input heldout_500.json, output answer.txt
from src.generator_demo import *
from tqdm import tqdm
import json
from types import SimpleNamespace

if __name__ == "__main__":
    parser_dict = {
        "resume_ckpt": "./ckpt/bart_finetuned.ckpt",
        "file_path": "./data/STRING.txt",
        "bart_version": "bart-base",
        "device": 0,
        "max_length": 512,
        "max_output_length": 1024,
        "min_output_length": 8,
        "eval_beams": 5,
        "no_repeat_ngram_size": 50,
        "num_return_sequences": 5
    }
    args = SimpleNamespace(**parser_dict)
    model = Generator(args)

    df = pd.read_json("./test/heldout_500.json")
    result_list = []
    for i,row in tqdm(df.iterrows()):
        results=model.generate_demo(row["question"])
        new_dict={"id":row['id'],"answer":model.final_answer,"entities":["<"+a+">" for a in model.current_entity]}
        print(row["question"])
        print(new_dict)
        print("------------------------")
        result_list.append(new_dict)

    with open("./test/answer.txt", 'w') as json_file:
        json.dump(result_list, json_file, indent=6)
    print("saving done!")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 19:26
# @Author  : Zhiruo Zhang
# @File    : generator_demo.py
# @Description : model for demo website, there are some modifications based on generator_main.py.
from transformers import BartTokenizer, BartForConditionalGeneration
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
import logging
import re
import difflib
from collections import OrderedDict
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from itertools import product
os.makedirs("./logs",exist_ok=True)
logging.basicConfig(filename=f'./logs/{datetime.now().strftime("%Y-%m-%d")}.log',level=logging.INFO,filemode='a')
logger = logging.getLogger(__name__)
import argparse
import time
import torch
from SPARQLWrapper import SPARQLWrapper, JSON
import traceback
import ast

class Generator:
    def __init__(self, args):
        self.args = args
        self.sparql_endpoint = "https://dblp-kg.ltdemos.informatik.uni-hamburg.de/sparql"
        self.sparql = SPARQLWrapper(self.sparql_endpoint)
        self.dblp_website="https://dblp.org"

        self.question=None

        self.prediction=None
        self.prediction_postprocessed =None
        self.info_dict=None

        self.entity_mapping=None
        self.string_mapping=None
        self.number = None

        self.templates=None
        self.templates_postprocessed=None

        self.final_query = None
        self.final_query_template = None
        self.final_query_template_postprocessed = None
        self.final_answer = None
        self.final_answer_dict = None

        self.current_number = None
        self.current_string = None
        self.current_entity = None

        self.updated_entity_mapping = None
        self.updated_string_mapping=None

        # load model
        self.load()
        logger.info("--------------------------------------------------------------READY--------------------------------------------------------------")

    def initialize(self,question, prediction, prediction_postprocessed, info_dict, entity_mapping, string_mapping, number,
                   templates, templates_postprocessed, final_query, final_query_template,
                   final_query_template_postprocessed, final_answer, final_answer_dict,current_number, current_string, current_entity,updated_entity_mapping,updated_string_mapping):
        self.question = question[0] if isinstance(question, tuple) else question
        self.prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        self.prediction_postprocessed = prediction_postprocessed[0] if isinstance(prediction_postprocessed, tuple) else prediction_postprocessed
        self.info_dict = info_dict[0] if isinstance(info_dict, tuple) else info_dict
        self.entity_mapping = entity_mapping[0] if isinstance(entity_mapping, tuple) else entity_mapping
        self.string_mapping = string_mapping[0] if isinstance(string_mapping, tuple) else string_mapping
        self.number = number[0] if isinstance(number, tuple) else number
        self.templates = templates[0] if isinstance(templates, tuple) else templates
        self.templates_postprocessed = templates_postprocessed[0] if isinstance(templates_postprocessed, tuple) else templates_postprocessed
        self.final_query = final_query[0] if isinstance(final_query, tuple) else final_query
        self.final_query_template = final_query_template[0] if isinstance(final_query_template, tuple) else final_query_template
        self.final_query_template_postprocessed = final_query_template_postprocessed[0] if isinstance( final_query_template_postprocessed, tuple) else final_query_template_postprocessed
        self.final_answer = final_answer[0] if isinstance(final_answer, tuple) else final_answer
        self.final_answer_dict = final_answer_dict[0] if isinstance(final_answer_dict, tuple) else final_answer_dict
        self.current_number = current_number[0] if isinstance(current_number, tuple) else current_number
        self.current_string = current_string[0] if isinstance(current_string, tuple) else current_string
        self.current_entity = current_entity[0] if isinstance(current_entity, tuple) else current_entity
        self.updated_entity_mapping=updated_entity_mapping [0] if isinstance(updated_entity_mapping, tuple) else updated_entity_mapping
        self.updated_string_mapping=updated_string_mapping[0] if isinstance(updated_string_mapping, tuple) else updated_string_mapping

    def load_from_checkpoint(self, model, resume_ckpt):
        logger.info(f"loading from checkpoint......path: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location="cpu")
        epoch = checkpoint["epoch"]
        step = checkpoint["global_step"]
        logger.info(
            f"number of executed epochs of finetuned model(starts from 1): {epoch}\nnumber of executed steps of finetuned model(starts from 0): {step}")
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("finetuned model loaded!")
        return model

    def load(self):
        self.sss, self.sss_s, self.vocab_dict, self.rel_d = self.prepare()
        self.string_list = []
        with open(self.args.file_path, "r") as file:
            self.string_list = [line.strip() for line in file.readlines()]
        self.string_dict = {key: key.lower() for key in self.string_list}
        logger.info("loading model and tokenizer")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/" + self.args.bart_version, add_prefix_space=True,
                                                       additional_special_tokens=list(self.vocab_dict.values()))
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/" + self.args.bart_version)
        bart_model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.load_from_checkpoint(bart_model, self.args.resume_ckpt)

        self.rank = torch.device("cuda:" + str(self.args.device) if torch.cuda.is_available() else "cpu")
        logger.info(f"device: {self.rank}")
        self.model.to(self.rank)
        self.model.eval()

    def convert_number(self, question):
        numbers = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10"
        }
        for nk, nv in numbers.items():
            if " in the last {} years".format(nk) in question:
                question = question.replace(" in the last {} years".format(nk), " in the last {} years".format(nv))
        return question

    def prepare(self):
        sss = [
            '<eid_33> <eid_28> <eid_50><eid_58><eid_10><eid_58><eid_29><eid_59><eid_59> <eid_51> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57> <eid_42> <eid_56> <eid_29> <eid_14> <eid_1> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_29> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_13> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_13> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_14> <eid_26> <eid_32><eid_58><eid_26> <eid_3> <eid_0><eid_59> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_20> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_29> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_34> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_6> <eid_53> <eid_18> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>',
            '<eid_47> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57>',
            '<eid_33> <eid_58><eid_46><eid_58><eid_30><eid_59> <eid_51> <eid_29><eid_59> <eid_56> <eid_33> <eid_58><eid_38><eid_58><eid_53><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_57> <eid_39> <eid_55> <eid_53> <eid_57>',
            '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_6> <eid_53> <eid_18> <eid_29> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_52> <eid_14> <eid_26> <eid_6> <eid_52> <eid_18> <eid_24> <eid_32> <eid_58><eid_52> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_20> <eid_52> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_53> <eid_14> <eid_52> <eid_32> <eid_58><eid_53> <eid_3> <eid_0><eid_59> <eid_6> <eid_53> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_18> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_49><eid_58><eid_10><eid_58><eid_29><eid_59><eid_59> <eid_51> <eid_29> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57>',
            '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_0> <eid_18> <eid_34> <eid_6> <eid_0> <eid_13> <eid_35> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_15> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_15> <eid_29> <eid_57> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59><eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_33> <eid_28> <eid_29> <eid_50><eid_58><eid_10><eid_58><eid_53><eid_59><eid_59> <eid_51> <eid_53> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_52> <eid_14> <eid_29> <eid_6> <eid_52> <eid_13> <eid_53> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_26> <eid_14> <eid_0> <eid_6> <eid_26> <eid_13> <eid_24> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_13> <eid_29> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_21> <eid_23> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_13> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_18> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_56> <eid_29> <eid_18> <eid_35> <eid_57> <eid_42> <eid_56> <eid_29> <eid_18> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_56> <eid_29> <eid_18> <eid_34> <eid_57> <eid_42> <eid_56> <eid_29> <eid_18> <eid_35> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_13> <eid_52> <eid_6> <eid_1> <eid_13> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_8> <eid_53> <eid_11> <eid_0> <eid_11> <eid_1><eid_59> <eid_51> <eid_29><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_12> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_12> <eid_29> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_52> <eid_6> <eid_1> <eid_15> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_7> <eid_53> <eid_11> <eid_0> <eid_11> <https://dblp.org/rec/conf/sigmetrics/GastH18><eid_59> <eid_51> <eid_29><eid_59> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_29> <eid_32><eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_35> <eid_6> <eid_0> <eid_13> <eid_34> <eid_6> <eid_0> <eid_16> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_17> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_13> <eid_53> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_54> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_53> <eid_14> <eid_29> <eid_6> <eid_53> <eid_13> <eid_54> <eid_57> <eid_39> <eid_55> <eid_54> <eid_57> <eid_41> <eid_55> <eid_45><eid_58><eid_54><eid_59> <eid_40> 1',
            '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_32> <eid_48> <eid_31> <eid_56> <eid_1> <eid_14> <eid_0> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_18> <eid_34> <eid_6> <eid_0> <eid_13> <eid_35> <eid_6> <eid_0> <eid_16> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_14> <eid_26> <eid_6> <eid_24> <eid_14> <eid_26> <eid_32> <eid_58><eid_24> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_0> <eid_18> <eid_35> <eid_6> <eid_0> <eid_13> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_12> <eid_34> <eid_6> <eid_0> <eid_13> <eid_29> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_32> <eid_58><eid_29> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57> <eid_42> <eid_56> <eid_0> <eid_14> <eid_29> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_26> <eid_32><eid_58><eid_26> <eid_3> <eid_0><eid_59> <eid_6> <eid_26> <eid_12> <eid_24> <eid_57>',
            '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_6> <eid_1> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_21> <eid_22> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_52><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_52> <eid_14> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_44><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_47> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_32> <eid_48> <eid_31> <eid_56> <eid_2> <eid_14> <eid_0> <eid_6> <eid_2> <eid_14> <eid_1> <eid_57> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_57>',
            '<eid_33> <eid_28> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_6> <eid_29> <eid_13> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_26> <eid_6> <eid_52> <eid_16> <eid_24> <eid_57>',
            '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_0> <eid_13> <eid_53> <eid_6> <eid_54> <eid_14> <eid_52> <eid_6> <eid_54> <eid_13> <eid_29> <eid_32> <eid_58><eid_29> <eid_3> <eid_53><eid_59> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_12> <eid_34> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_29> <eid_14> <eid_52> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_1> <eid_6> <eid_0> <eid_18> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_17> <eid_29> <eid_57>',
            '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_14> <eid_1> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_12> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_18> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_18> <eid_29> <eid_57> <eid_57>',
            '<eid_47> <eid_56> <eid_1> <eid_14> <eid_0> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_53> <eid_6> <eid_53> <eid_12> <eid_34> <eid_57>',
            '<eid_33> <eid_58><eid_46><eid_58><eid_30><eid_59> <eid_51> <eid_29><eid_59> <eid_56> <eid_33> <eid_58><eid_38><eid_58><eid_53><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_15> <eid_53> <eid_57> <eid_39> <eid_55> <eid_53> <eid_57>',
            '<eid_47> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_32> <eid_48> <eid_31> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_1> <eid_14> <eid_52> <eid_32> <eid_58><eid_1> <eid_3> <eid_0><eid_59> <eid_57> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_0> <eid_14> <eid_29> <eid_6> <eid_29> <eid_12> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_0> <eid_13> <eid_29> <eid_57> <eid_42> <eid_56> <eid_1> <eid_13> <eid_29> <eid_57> <eid_57>',
            '<eid_33> <eid_58><eid_38><eid_58><eid_28> <eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_14> <eid_1> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_15> <eid_52> <eid_6> <eid_1> <eid_15> <eid_53> <eid_6> <eid_36><eid_58><eid_37><eid_58><eid_52> <eid_7> <eid_53> <eid_11> <eid_0> <eid_11> <eid_1><eid_59> <eid_51> <eid_29><eid_59> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_19> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_18> <eid_26> <eid_6> <eid_52> <eid_16> <eid_24> <eid_57>',
            '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_34> <eid_32> <eid_48> <eid_31> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_18> <eid_34> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_56> <eid_29> <eid_14> <eid_0> <eid_57> <eid_42> <eid_56> <eid_29> <eid_14> <eid_1> <eid_57> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_21> <eid_29> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_0> <eid_14> <eid_52> <eid_6> <eid_52> <eid_20> <eid_29> <eid_57>',
            '<eid_33> <eid_58><eid_25><eid_58><eid_29> <eid_9><eid_59> <eid_51> <eid_29><eid_59> <eid_30> <eid_43> <eid_56> <eid_33> <eid_28> <eid_29> <eid_58><eid_38><eid_58><eid_29><eid_59> <eid_51> <eid_30><eid_59> <eid_43> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_29> <eid_57> <eid_39> <eid_55> <eid_29> <eid_57> <eid_41> <eid_55> <eid_45><eid_58><eid_30><eid_59> <eid_40> 1',
            '<eid_33> <eid_28> <eid_26> <eid_24> <eid_43> <eid_56> <eid_0> <eid_18> <eid_26> <eid_6> <eid_0> <eid_13> <eid_24> <eid_57>',
            '<eid_47> <eid_56> <eid_52> <eid_14> <eid_0> <eid_6> <eid_52> <eid_13> <eid_53> <eid_6> <eid_32><eid_58><eid_53> <eid_4> <eid_5><eid_59> <eid_6> <eid_52> <eid_18> <eid_34> <eid_57>',
            '<eid_33> <eid_28> <eid_29> <eid_43> <eid_56> <eid_29> <eid_14> <eid_0> <eid_6> <eid_29> <eid_18> <eid_34> <eid_57>']

        initial = ["<topic1>", "<topic2>", "<topic3>"] + ["<isnot>", "<within>", "<num>", "<dot>", "<dayu>", "<xiaoyu>",
                                                          "<comma_sep>", "<is_int>", "<comma>"] + [
                      "<primaryAffiliation>", "<yearOfPublication>", "<authoredBy>", "<numberOfCreators>", "<title>",
                      "<webpage>", "<publishedIn>", "<wikidata>", "<orcid>", "<bibtexType>", "<Inproceedings>",
                      "<Article>"]
        extra = ['?secondanswer', 'GROUP_CONCAT', '?firstanswer', 'separator', 'DISTINCT', '?answer', '?count',
                 'EXISTS', 'FILTER', 'SELECT', 'STRING1', 'STRING2', 'BIND', 'IF', 'COUNT', 'GROUP', 'LIMIT', 'ORDER',
                 'UNION', 'WHERE', 'DESC', 'ASC', 'AVG', 'ASK', 'NOT', 'MAX', 'MIN', 'AS', '?x', '?y', '?z', 'BY', "{",
                 "}", "(", ")"]
        vocab = initial + extra
        vocab_dict = {}
        for i, text in enumerate(vocab):
            vocab_dict[text] = '<eid_' + str(i) + '>'
        sss_s = [s.replace(vocab_dict["<num>"], "NUMBER").replace(vocab_dict["STRING1"], "STRING1").replace(
            vocab_dict["STRING2"],
            "STRING2").replace(
            vocab_dict["<topic1>"], "TOPIC1").replace(vocab_dict["<topic2>"], "TOPIC2").replace(vocab_dict["<topic3>"],
                                                                                                "TOPIC3").replace(
            " ", "") for s in sss]

        rel_d1 = {'<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>',
                  '<https://dblp.org/rdf/schema#wikidata>': '<wikidata>',
                  '<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>',
                  '<https://dblp.org/rdf/schema#webpage>': '<webpage>',
                  '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>',
                  '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>',
                  '<https://dblp.org/rdf/schema#title>': '<title>',
                  '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>'}
        rel_d2 = {'<https://dblp.org/rdf/schema#wikidata>': '<wikidata>',
                  '<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>',
                  '<https://dblp.org/rdf/schema#webpage>': '<webpage>',
                  '<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>',
                  '<https://dblp.org/rdf/schema#orcid>': '<orcid>',
                  '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>',
                  '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>',
                  '<https://dblp.org/rdf/schema#title>': '<title>',
                  '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>',
                  '<https://dblp.org/rdf/schema#bibtexType>': '<bibtexType>',
                  '<http://purl.org/dc/terms/bibtexType>': '<bibtexType>',
                  '<http://purl.org/net/nknouf/ns/bibtex#Article>': '<Article>',
                  '<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>': '<Inproceedings>'}
        rel_d3 = {'<https://dblp.org/rdf/schema#primaryAffiliation>': '<primaryAffiliation>',
                  '<https://dblp.org/rdf/schema#authoredBy>': '<authoredBy>',
                  '<https://dblp.org/rdf/schema#orcid>': '<orcid>',
                  '<https://dblp.org/rdf/schema#webpage>': '<webpage>',
                  '<https://dblp.org/rdf/schema#wikidata>': '<wikidata>',
                  '<https://dblp.org/rdf/schema#publishedIn>': '<publishedIn>',
                  '<https://dblp.org/rdf/schema#yearOfPublication>': '<yearOfPublication>',
                  '<https://dblp.org/rdf/schema#title>': '<title>',
                  '<https://dblp.org/rdf/schema#numberOfCreators>': '<numberOfCreators>',
                  '<https://dblp.org/rdf/schema#bibtexType>': '<bibtexType>',
                  '<http://purl.org/dc/terms/bibtexType>': '<bibtexType>',
                  '<http://purl.org/net/nknouf/ns/bibtex#Inproceedings>': '<Inproceedings>',
                  '<http://purl.org/net/nknouf/ns/bibtex#Article>': '<Article>'}
        rel_d = {**rel_d1, **rel_d2, **rel_d3}
        rel_d = {v: [k] for k, v in rel_d.items()}
        rel_d["<bibtexType>"] = rel_d["<bibtexType>"] + ["<https://dblp.org/rdf/schema#bibtexType>"]

        return sss, sss_s, vocab_dict, rel_d

    def inference(self, question):
        '''
        :param question:
        :return: prediction,prediction_postprocessed
        '''
        encoding = self.tokenizer(question, max_length=int(self.args.max_length), return_tensors='pt', truncation=True,
                                  add_prefix_space=True, padding="max_length")
        encoding.to(self.rank)
        with torch.no_grad():
            output = self.model.generate(
                **encoding,
                use_cache=True,
                num_beams=self.args.eval_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=self.tokenizer.pad_token_id,
                max_length=self.args.max_output_length,
                min_length=self.args.min_output_length,
                early_stopping=True,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            )
            prediction = self.tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=True)
            prediction = prediction.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("  ",
                                                                                                        " ").strip().lower()
            prediction_postprocessed = prediction
            for k, v in reversed(OrderedDict(self.vocab_dict).items()):
                prediction_postprocessed = prediction_postprocessed.replace(v, k)
        return prediction, prediction_postprocessed

    def get_url_by_page(self, label):
        max_retries = 5
        retry_delay = 0.1
        url = f"{self.dblp_website}/search/author?q=" + label
        for retry in range(max_retries):
            response = requests.get(url, headers={'Connection': 'close'}, verify=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                label_element = soup.select('ul.result-list > li > a')
                result_list = []
                if label_element:
                    for element in label_element:
                        result_list.append(element.get('href'))
                        if len(result_list) >= 10:  # top 10
                            return result_list
            elif response.status_code == 429:
                if retry == 0:
                    ttt = retry_delay
                elif retry == 1:
                    ttt = 1
                elif retry == 2:
                    ttt = 30
                else:
                    ttt = 60
                time.sleep(ttt)
        return []

    def get_url(self, label_ori):
        label = label_ori.replace("'", "&apos;")
        label_ori = label_ori.replace("'", "&apos;")
        for tryt in label.split(" "):
            if tryt.strip().startswith("000") and tryt.strip().isdigit():
                label = label.replace(" " + tryt.strip(), "")

        author_url = f"{self.dblp_website}/search/author/api?q=" + label + "&h=1000&format=json"
        pub_url = f"{self.dblp_website}/search/publ/api?q=" + label + "&h=1000&format=json"

        max_retries = 5
        retry_delay = 0.1

        for retry in range(max_retries):
            response = requests.get(author_url, headers={'Connection': 'close'}, verify=False)
            url = author_url
            if response.status_code == 200:
                data = response.json()
                lll = {}
                if 'hit' in data['result']['hits'].keys():
                    for o in data['result']['hits']['hit']:
                        if re.sub(r'\d', '', o["info"]["author"]) not in lll.keys():
                            lll[re.sub(r'\d', '', o["info"]["author"])] = []
                        lll[re.sub(r'\d', '', o["info"]["author"])].append(o["info"]["url"])
                    if int(data['result']['hits']['@total']) <= 1000:
                        tmp = difflib.get_close_matches(label_ori, list(lll.keys()), n=10, cutoff=0.01)
                        result_list = []
                        if len(tmp) > 0:
                            for t in tmp:
                                result_list.extend(lll[t])
                                if len(result_list) >= 10:
                                    return result_list
                            return result_list
                        else:
                            response = requests.get(pub_url)
                            url = pub_url
                    else:
                        return self.get_url_by_page(label)
                else:
                    response = requests.get(pub_url, headers={'Connection': 'close'}, verify=False)
                    url = pub_url
            if response.status_code == 200:
                data = response.json()
                lll = {}
                if 'hit' in data['result']['hits'].keys():
                    for o in data['result']['hits']['hit']:
                        if o["info"]["title"] not in lll.keys():
                            lll[o["info"]["title"]] = []

                        lll[o["info"]["title"]].append(o["info"]["url"])
                    tmp = difflib.get_close_matches(label_ori, list(lll.keys()), n=10, cutoff=0.01)
                    result_list = []
                    if len(tmp) > 0:
                        for t in tmp:
                            result_list.extend(lll[t])
                            if len(result_list) >= 10:
                                return result_list
                        return result_list
            elif response.status_code == 429:
                if retry == 0:
                    ttt = retry_delay
                elif retry == 1:
                    ttt = 1
                elif retry == 2:
                    ttt = 30
                else:
                    ttt = 60
                time.sleep(ttt)
        return []

    def get_venue_url(self, label):
        venue_url = f"{self.dblp_website}/search/venue/api?q=" + label + "&h=1000&format=json"

        max_retries = 5
        retry_delay = 0.1

        try:
            for retry in range(max_retries):
                response = requests.get(venue_url, headers={'Connection': 'close'}, verify=False)
                if response.status_code == 200:
                    lll = []
                    data = response.json()
                    if 'hit' in data['result']['hits'].keys():
                        for o in data['result']['hits']['hit']:
                            if o["info"]["acronym"]:
                                if o["info"]["acronym"] not in lll:
                                    lll.append(o["info"]["acronym"])
                            elif o["info"]["venue"]:
                                if o["info"]["venue"] not in lll:
                                    lll.append(o["info"]["venue"])
                    return lll
                elif response.status_code == 429:
                    if retry == 0:
                        ttt = retry_delay
                    elif retry == 1:
                        ttt = 1
                    elif retry == 2:
                        ttt = 30
                    else:
                        ttt = 60
                    time.sleep(ttt)
            return []
        except Exception as e:
            logger.info(f"ERROR WHEN invoking get_venue_url(): {e}")
            return []

    def get_fullname(self, s):
        tmp = []
        string_list = self.string_dict.values()
        for one in string_list:
            if one.startswith(s):
                tmp.append(one)

        if len(tmp) == 0:
            lower_strings = difflib.get_close_matches(s.lower(), string_list, n=10, cutoff=0.8)
        else:
            lower_strings = difflib.get_close_matches(s.lower(), tmp, n=10, cutoff=0.8)
        strings2 = [key for element in lower_strings for key, value in self.string_dict.items() if value == element]

        if len(strings2)>0:
            if len(tmp) == 0:
                lower_strings = difflib.get_close_matches(s.lower(), string_list, n=10, cutoff=0.01)
            else:
                lower_strings = difflib.get_close_matches(s.lower(), tmp, n=10, cutoff=0.01)
            return [key for element in lower_strings for key, value in self.string_dict.items() if value == element]
        else:
            strings = self.get_venue_url(s)
            if len(tmp) == 0:
                lower_strings = difflib.get_close_matches(s.lower(), string_list, n=10, cutoff=0.01)
            else:
                lower_strings = difflib.get_close_matches(s.lower(), tmp, n=10, cutoff=0.01)
            strings2 = [key for element in lower_strings for key, value in self.string_dict.items() if value == element]
            for s in strings2:
                if s not in strings:
                    strings.append(s)
            return strings

    def extract_info(self, prediction):
        for i in self.vocab_dict.values():
            prediction = prediction.replace(i, " # ")
        prediction = prediction.replace("   ", " ").replace("  ", " ").strip().encode('utf-8').decode('utf-8')

        phrase = re.findall(r"(?:(?![ #]+)[\w'-\\:/,ÃãÍÇíçÑñÜü ]+ ?)+", prediction)

        if phrase and len(phrase) > 0:
            phrase = [p.strip() for p in phrase]
        info_dict = {"entity": [], "number": [], "string": []}

        for p in phrase:
            if p.startswith("'") and p.endswith("'"):
                info_dict["string"].append(p)
            elif p.isdigit():
                info_dict["number"].append(p)
            elif len(p) > 0:
                info_dict["entity"].append(p)
        entity_mapping = {}
        for e in info_dict["entity"]:
            url = self.get_url(e)
            if len(url) > 0:
                entity_mapping[e] = url
            else:
                entity_mapping[e] = [e]
        string_mapping = {}
        for s in info_dict["string"]:
            strings = self.get_fullname(s.strip("'").strip('"'))
            if len(strings) > 0:
                string_mapping[s ] = strings
            else:
                string_mapping[s] = [s]
        return info_dict, entity_mapping, string_mapping

    def get_template(self, prediction, info_dict, cans=5):
        for entity in info_dict['entity']:
            prediction = prediction.replace(entity, "TOPIC")
        for number in info_dict['number']:
            prediction = prediction.replace(" " + number, " NUMBER")
            prediction = prediction.replace(number + " ", "NUMBER ")
        for string in info_dict['string']:
            prediction = prediction.replace(string, "STRING")
        templates = []
        for i in difflib.get_close_matches(prediction.replace(" ",""), self.sss_s, n=cans, cutoff=0.01):
            indexes_of_a = [index for index, value in enumerate(self.sss_s) if value == i]
            for one in indexes_of_a:
                if self.sss[one] not in templates:
                    nc = 0
                    if self.vocab_dict["<topic3>"] in self.sss[one]:
                        nc += 1
                    if self.vocab_dict["<topic2>"] in self.sss[one]:
                        nc += 1
                    if self.vocab_dict["<topic1>"] in self.sss[one]:
                        nc += 1
                    num_entities = nc
                    if self.vocab_dict["STRING2"] in self.sss[one]:
                        num_string = 2
                    elif self.vocab_dict["STRING1"] in self.sss[one]:
                        num_string = 1
                    else:
                        num_string = 0
                    if self.vocab_dict["<num>"] in self.sss[one]:
                        num_num = 1
                    else:
                        num_num = 0
                    # validate templates
                    if len(info_dict['entity']) >= num_entities and len(info_dict["string"]) >= num_string and len(
                            info_dict["number"]) >= num_num:
                        templates.append(self.sss[one])
        templates_postprocessed = []
        for t in templates:
            for k, v in reversed(OrderedDict(self.vocab_dict).items()):
                t = t.replace(v, k)
            templates_postprocessed.append(self.sparql_format(t))
            # templates_postprocessed.append(t)
        return templates, templates_postprocessed

    def to_query(self, template, templates_postprocessed, updated_entity_mapping, updated_string_mapping, number: list):
        # templates_postprocessed=self.reset_color_format(templates_postprocessed)

        if self.vocab_dict["STRING2"] in template:
            num_string = 2
        elif self.vocab_dict["STRING1"] in template:
            num_string = 1
        else:
            num_string = 0
        if self.vocab_dict["<num>"] in template:
            num_num = 1
        else:
            num_num = 0

        special = {"<isnot>": "!=", "<dot>": ".", "?answer <comma_sep>": "?answer; separator=', '",
                   "<is_int>": "xsd:integer", "<xiaoyu>": "<", "<dayu>": ">", "<comma>": ",",
                   "<within>": "> YEAR(NOW())-"}
        query_l = []
        if len(updated_string_mapping)==0:
            current_string=["'"+self.string_mapping[label][0].strip("'").strip('"')+"'" for label in self.info_dict["string"]]
        else:
            current_string = ["'"+self.updated_string_mapping[label][0].strip("'").strip('"')+"'" for label in self.info_dict["string"]]

        # In DBLP-QuAD, there are SPARQL queries wrongly written, here we need to do some modifications to correct the SPARQL queries
        # ?y <within> <num> ---> xsd:integer(?y) <within> <num>
        templates_postprocessed=templates_postprocessed.replace("?y <within> <num>","xsd:integer(?y) <within> <num>")

        if num_string == 2:
            if templates_postprocessed.index("STRING1") < templates_postprocessed.index("STRING2"):
                templates_postprocessed = templates_postprocessed.replace("STRING1", current_string[0])
                templates_postprocessed = templates_postprocessed.replace("STRING2", current_string[1])
            else:
                templates_postprocessed = templates_postprocessed.replace("STRING1", current_string[1])
                templates_postprocessed = templates_postprocessed.replace("STRING2", current_string[0])
        elif num_string == 1:
            templates_postprocessed = templates_postprocessed.replace("STRING1", current_string[0])
        if num_num == 1:
            templates_postprocessed = templates_postprocessed.replace("<num>", number[0])
        # replace relation token with complete relation token
        tmp = [templates_postprocessed, templates_postprocessed]
        for k, v in self.rel_d.items():
            if k in tmp[0]:
                if len(v) > 1:
                    tmp[0] = tmp[0].replace(k, v[0])
                    tmp[1] = tmp[1].replace(k, v[1])
                else:
                    tmp[0] = tmp[0].replace(k, v[0])
                    tmp[1] = tmp[1].replace(k, v[0])
        if tmp[0] == tmp[1]:
            tmp = list(set(tmp))
        # replace entities
        mapp_df=None
        if len(updated_entity_mapping)==0:
            updated_entity_mapping=self.entity_mapping
        for t in tmp:
            topics = re.findall(r'<topic\d+>', t)
            mapp = {}
            for topic in topics:
                mapp[topic] = ""
            for i in range(len(mapp.keys())):
                mapp[list(mapp.keys())[i]]=updated_entity_mapping[self.info_dict["entity"][i]]
            combinations = list(product(*mapp.values()))
            if mapp_df is None:
                mapp_df = pd.DataFrame(combinations, columns=mapp.keys())
            else:
                mapp_df=pd.concat([mapp_df,pd.DataFrame(combinations, columns=mapp.keys())],axis=0, ignore_index=True)

            for k, v in special.items():
                t = t.replace(k, v)

            cols = list(mapp.keys())
            for i, row in mapp_df.iterrows():
                query_tmp = t
                for col in cols:
                    query_tmp = query_tmp.replace(col, "<" + row[col] + ">")
                query_l.append(query_tmp)

        return query_l,mapp_df

    def get_answer(self, x: dict):
        l = []
        d={}
        if "results" in x.keys():
            for var in x["head"]["vars"]:
                d[var]=[]
            ll = list(x["results"]["bindings"])
            for one in ll:
                for k,v in one.items():
                    d[k].append(v['value'])
                    l.append(v["value"])
            return l,d
        else:
            if "boolean" in x.keys():
                l.append(x["boolean"])  # True or False
                d["boolean"]=[str(x["boolean"])]
            return l,d

    def do_query(self, query):
        try:
            self.sparql.setQuery(f"""
                    {query}
                    """)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            return results
        except Exception as e:
            logger.info(f"ERROR WHEN QUERYING: {e}")
            # traceback.print_exc()
            return {}

    def generate_demo(self,question):
        # try:
        self.question=question
        logger.info(f"INPUT QUESTION:\n{self.question}")
        # get inference
        self.prediction, self.prediction_postprocessed = self.inference(self.convert_number(question))
        logger.info(f"LOGICAL FORM:\n{self.prediction_postprocessed}")
        # extract info
        self.info_dict, self.entity_mapping, self.string_mapping = self.extract_info(self.prediction)
        self.number=self.info_dict["number"]
        self.updated_entity_mapping=self.entity_mapping
        self.updated_string_mapping = self.string_mapping
        logger.info(f"INFO DICT:\n{self.info_dict}\nTOPIC:\n{self.entity_mapping}\nSTRING:\n{self.string_mapping}\nNUMBER:\n{self.number}")
        # get templates
        self.templates, self.templates_postprocessed = self.get_template(self.prediction, self.info_dict)
        logger.info(f"CANDIDATE TEMPLATES:\n{self.templates_postprocessed}")

        self.final_query=""
        self.final_query_template=""
        self.final_query_template_postprocessed = ""
        self.final_answer=[]
        self.final_answer_dict = {}
        self.current_number=self.number
        self.current_string = [self.updated_string_mapping[label][0] for label in self.info_dict["string"]]
        self.current_entity=[]

        for i in range(len(self.templates)):
            ql,mapp_df= self.to_query(self.templates[i], self.templates_postprocessed[i], self.updated_entity_mapping, self.updated_string_mapping,self.current_number)
            if ql and len(ql) > 0:
                for ind in range(len(ql)):
                    query=ql[ind]
                    logger.info(f"QUERYING:\n{query}")
                    result = self.do_query(query)
                    if self.final_query is None or len(self.final_query)==0:
                        self.final_query = query
                        self.final_query_template = self.templates[i]
                        self.final_query_template_postprocessed = self.templates_postprocessed[i]
                        if mapp_df is not None and len(mapp_df)>0:
                            self.current_entity = mapp_df.iloc[ind].tolist()

                    if "results" in result.keys():
                        if len(result["results"]["bindings"]) != 0:
                            self.final_answer,self.final_answer_dict = self.get_answer(result)
                            if len(self.final_answer) > 0:
                                self.final_query = query
                                self.final_query_template = self.templates[i]
                                self.final_query_template_postprocessed = self.templates_postprocessed[i]
                                if mapp_df is not None and len(mapp_df) > 0:
                                    self.current_entity = mapp_df.iloc[ind].tolist()
                                break
                    elif "boolean" in result.keys():
                        self.final_answer, self.final_answer_dict = self.get_answer(result)
                        if len(self.final_answer) > 0:
                            self.final_query = query
                            self.final_query_template = self.templates[i]
                            self.final_query_template_postprocessed = self.templates_postprocessed[i]
                            if mapp_df is not None and len(mapp_df) > 0:
                                self.current_entity = mapp_df.iloc[ind].tolist()
                            break
                if self.final_answer and len(self.final_answer) > 0:
                    break
        self.prediction_postprocessed = self.sparql_format(self.prediction_postprocessed) # format logical form
        logger.info(f"FINAL TEMPLATE:\n{self.final_query_template_postprocessed}")
        logger.info(f"CURRENT ENTITY:\n{self.current_entity}")
        if len(self.current_string) > 0:
            logger.info(f"CURRENT STRING:\n{self.current_string}")
        if len(self.current_number) > 0:
            logger.info(f"CURRENT NUMBER:\n{self.current_number}")
        logger.info(f"FINAL QUERY:\n{self.final_query}")
        if self.final_answer_dict and len(self.final_answer_dict) > 0:
            # logger.info(f"FINAL ANSWER:\n{self.final_answer}")
            logger.info(f"FINAL ANSWER DICT:\n{self.final_answer_dict}")
        else:
            logger.info("NO RESULT FOR THIS QUESTION!")
        #
        # except Exception as e:
        #     logger.info(f"ERROR!!!——————{e}")
        #     traceback.print_exc()

        return self.return_all()

    def update_template(self,updated_template:str):
        if updated_template in self.templates_postprocessed:
            self.final_query_template = self.templates[self.templates_postprocessed.index(updated_template)]
            self.final_query_template_postprocessed = updated_template
        else:
            index=-1
            for i in range(len(self.templates_postprocessed)):
                if self.templates_postprocessed[i].replace("\r", " ").replace("\n","").replace(" ","")==updated_template.replace("\r", " ").replace("\n","").replace(" ",""):
                    index=i
                    break
            if index>=0:
                self.final_query_template = self.templates[index]
                self.final_query_template_postprocessed = self.templates_postprocessed[index]
            else:
                raise Exception(f"update_template: UPDATED TEMPLATE: {updated_template} NOT IN LIST {str(self.templates_postprocessed)}")

        queries,mapp_df = self.to_query(self.final_query_template, self.final_query_template_postprocessed, self.updated_entity_mapping, self.updated_string_mapping,self.current_number)
        logger.info(f"UPDATING TEMPLATE TO:\n{self.final_query_template_postprocessed}")
        if mapp_df is not None and len(mapp_df) > 0:
            self.current_entity = mapp_df.iloc[0].tolist()
        if queries and len(queries)>0:
            self.final_query=queries[0]
            logger.info(f"UPDATING QUERY TO:\n{self.final_query}")

    def update_entity(self,updated_entity:dict):
        self.updated_entity_mapping=updated_entity
        queries, mapp_df = self.to_query(self.final_query_template, self.final_query_template_postprocessed,
                                         self.updated_entity_mapping, self.updated_string_mapping, self.current_number)
        logger.info(f"UPDATING ENTITY MAPPING TO:\n{self.updated_entity_mapping}")
        if mapp_df is not None and len(mapp_df) > 0:
            self.current_entity = mapp_df.iloc[0].tolist()
        if queries and len(queries) > 0:
            self.final_query = queries[0]
            logger.info(f"UPDATING QUERY TO:\n{self.final_query}")

    def update_string(self,updated_string:dict):
        self.updated_string_mapping=updated_string
        self.current_string = [self.updated_string_mapping[label][0] for label in self.info_dict["string"]] #the first one
        queries, mapp_df = self.to_query(self.final_query_template, self.final_query_template_postprocessed,
                                         self.updated_entity_mapping, self.updated_string_mapping, self.current_number)
        logger.info(f"UPDATING STRING MAPPING TO:\n{self.updated_string_mapping}")

        if mapp_df is not None and len(mapp_df) > 0:
            self.current_entity = mapp_df.iloc[0].tolist()

        if queries and len(queries) > 0:
            self.final_query = queries[0]
            logger.info(f"UPDATING QUERY TO:\n{self.final_query}")

    def update_number(self,updated_number:int):
        self.current_number=[str(updated_number)]
        queries, mapp_df = self.to_query(self.final_query_template, self.final_query_template_postprocessed,
                                         self.updated_entity_mapping, self.updated_string_mapping, self.current_number)
        logger.info(f"UPDATING NUMBER TO:\n{self.current_number}")
        if mapp_df is not None and len(mapp_df) > 0:
            self.current_entity = mapp_df.iloc[0].tolist()

        if queries and len(queries) > 0:
            self.final_query = queries[0]
            logger.info(f"UPDATING QUERY TO:\n{self.final_query}")

    def single_query(self):
        logger.info(f"FINAL QUERY:\n{self.final_query}")
        result = self.do_query(self.final_query)
        self.final_answer, self.final_answer_dict = self.get_answer(result)
        # logger.info(f"FINAL ANSWER:\n{self.final_answer}")
        logger.info(f"FINAL ANSWER DICT:\n{self.final_answer_dict}")

    def return_all(self):
        return self.question, self.prediction, self.prediction_postprocessed, self.info_dict, self.entity_mapping, self.string_mapping, self.number, self.templates, self.templates_postprocessed,self.final_query ,self.final_query_template ,self.final_query_template_postprocessed ,self.final_answer ,self.final_answer_dict ,self.current_number,self.current_string ,self.current_entity,self.updated_entity_mapping,self.updated_string_mapping

    def sparql_format(self,sparql):
        try:
            split_pattern = re.compile(r'({|}|UNION|<dot>)')
            splits = filter(None, split_pattern.split(sparql))
            result = []
            for part in splits:
                if len(part.strip("\n").strip()) > 0:
                    temp = [r.strip() for r in result]
                    if "}" in part.strip("\n").strip():
                        indent = temp.count("{") - temp.count("}") - 1
                    else:
                        indent = temp.count("{") - temp.count("}")
                    result.append("    " * indent + part.strip("\n").strip())
            string=""
            for i in range(len(result)):
                if i<len(result)-1:
                    if result[i].strip()!="<dot>":
                        string += result[i]
                        if result[i+1].strip()=="<dot>":
                            string+= " <dot>"
                        string+="\n"
                else:
                    string+=result[i]
            # return self.color_format(string)
            return string
        except Exception as e:
            logger.info(f"ERROR WHEN INVOKING sparql_format: {e}")
            traceback.print_exc()
            return sparql

    def color_format(self,sparql):
        green = '\033[32m'  # for variables
        red = '\033[91m'  # for keywords
        yellow = '\033[93m'  # for strings/numbers
        blue = '\033[94m'  # for entities and relations
        reset = '\033[0m'  # reset color

        variable_regex = r'\?(secondanswer|firstanswer|answer|count|x|y|z)'
        keyword_regex = r'(SELECT|DISTINCT|WHERE|ASK|OPTIONAL|GROUP|COUNT|ORDER|LIMIT|OFFSET|MIN|MAX|AVG|UNION|FILTER|NOT|EXISTS|AS|BY|DESC|ASC|BIND|IF|<xiaoyu>|<dayu>|<is_int>|<within>|<isnot>|GROUP_CONCAT|<comma_sep>|<comma>|<dot>)'
        string_number_regex = r'(STRING1|STRING2|<num>)'
        entity_relation_regex = r'(<topic1>|<topic2>|<topic3>|<authoredBy>|<wikidata>|<primaryAffiliation>|<webpage>|<yearOfPublication>|<publishedIn>|<title>|<numberOfCreators>|<orcid>|<bibtexType>|<Article>|<Inproceedings>)'

        try:
            r1 = re.findall(variable_regex, sparql)
            r2 = re.findall(keyword_regex, sparql)
            r3 = re.findall(string_number_regex, sparql)
            r4 = re.findall(entity_relation_regex, sparql)
            if r1:
                for r in r1:
                    newr = "?" + r
                    sparql = sparql.replace(newr, f'{green}{newr}{reset}')
            if r2:
                for r in r2:
                    sparql = sparql.replace(r, f'{red}{r}{reset}')
            if r3:
                for r in r3:
                    sparql = sparql.replace(r, f'{yellow}{r}{reset}')
            if r4:
                for r in r4:
                    sparql = sparql.replace(r, f'{blue}{r}{reset}')
            return sparql
            # return self.reset_color_format(sparql)
        except Exception as e:
            logger.info(f"ERROR WHEN INVOKING color_format: {e}")
            traceback.print_exc()
            # return self.reset_color_format(sparql)
            return sparql

    def reset_color_format(self, sparql):
        reset = '\033[0m'  # reset color
        colorized_regex = re.compile(r'\033\[\d+m')
        sparql = colorized_regex.sub('', sparql)
        return sparql

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument("--default_dir", type=str, default="./", dest="default_dir")
    parser.add_argument("--file_path", type=str, default="data/STRING.txt", dest="file_path")
    parser.add_argument("--resume_ckpt", type=str, default="ckpt/bart_finetuned.ckpt", dest="resume_ckpt")
    parser.add_argument("--bart_version", type=str, help="bart-base or bart-large", default="bart-base",
                        dest="bart_version")
    parser.add_argument("--device", help="gpu device number", default=0, type=int, dest="device")
    parser.add_argument("--max_length", type=int, help="max_length when encoding", default=512, dest="max_length")
    parser.add_argument("--max_output_length", type=int, default=1024,
                        help="max_output_length for generation", dest="max_output_length")
    parser.add_argument("--min_output_length", type=int, default=8,
                        help="min_output_l for generation", dest="min_output_length")
    parser.add_argument("--eval_beams", type=int, default=5, help="beam size for inference when testing/validating",
                        dest="eval_beams")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=50,
                        help="no_repeat_ngram_size when generating predictions",
                        dest="no_repeat_ngram_size")
    parser.add_argument("--num_return_sequences", type=int, default=5,dest="num_return_sequences")

    args = parser.parse_args()
    args.resume_ckpt = os.path.join(args.default_dir, args.resume_ckpt)
    args.file_path = os.path.join(args.default_dir, args.file_path)

    nlqxform = Generator(args)

    while True:
        question = input("please input your question: ")
        if question == "BYE":
            exit()
        else:
            try:
                nlqxform.generate_demo(question)
                while True:
                    choice = input(
                        "*******\nCHOOSE 0 TO END THIS QUESTION. 1: update template(str) 2:update entity(dict) 3: update string(dict) 4:update number(int) 5:do query:\n")
                    if choice.isdigit():
                        if int(choice) == 0:
                            break
                        elif int(choice) == 1:
                            template = input("please input selected template: ")
                            nlqxform.update_template(template)
                        elif int(choice) == 2:
                            entity = input("please input entity dict: ")
                            nlqxform.update_entity(ast.literal_eval(entity))
                        elif int(choice) == 3:
                            string = input("please input string dict: ")
                            nlqxform.update_string(ast.literal_eval(string))
                        elif int(choice) == 4:
                            number = input("please input number: ")
                            nlqxform.update_number(int(number))
                        elif int(choice) == 5:
                            print("run query............")
                            nlqxform.single_query()
                    else:
                        break
            except Exception as e:
                logger.info(f"ERROR!!!——————{e}")
                traceback.print_exc()
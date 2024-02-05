#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 19:36
# @Author  : Zhiruo Zhang
# @File    : app.py
# @Description : demo

from flask import Flask, render_template, request, jsonify, session, make_response
from src.generator_demo import *
from types import SimpleNamespace
from pydantic import BaseModel
import json

app = Flask(__name__)
app.secret_key = '123'

class Info(BaseModel):
    question: str

    prediction: str
    prediction_postprocessed: str
    info_dict: dict

    entity_mapping: dict
    string_mapping: dict
    number: list

    templates: list
    templates_postprocessed: list

    final_query: str
    final_query_template: str
    final_query_template_postprocessed: str
    final_answer: list
    final_answer_dict: dict

    current_number: list
    current_string: list
    current_entity: list

    updated_entity_mapping:dict
    updated_string_mapping:dict

    def __init__(self, question, prediction, prediction_postprocessed, info_dict, entity_mapping, string_mapping,
                 number, templates, templates_postprocessed, final_query, final_query_template,
                 final_query_template_postprocessed, final_answer,final_answer_dict, current_number, current_string,
                 current_entity, updated_entity_mapping,updated_string_mapping):
        super().__init__(
            question=question,
            prediction=prediction,
            prediction_postprocessed=prediction_postprocessed,
            info_dict=info_dict,
            entity_mapping=entity_mapping,
            string_mapping=string_mapping,
            number=number,
            templates=templates,
            templates_postprocessed=templates_postprocessed,
            final_query=final_query,
            final_query_template=final_query_template,
            final_query_template_postprocessed=final_query_template_postprocessed,
            final_answer=final_answer,
            final_answer_dict=final_answer_dict,
            current_number=current_number,
            current_string=current_string,
            current_entity=current_entity,
            updated_entity_mapping=updated_entity_mapping,
            updated_string_mapping=updated_string_mapping
        )

parser_dict = {
    "resume_ckpt": "./src/ckpt/bart_finetuned.ckpt",
    "file_path": "./src/data/STRING.txt",
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

@app.route('/', methods=['GET', 'POST'])
def index():
    info = None
    if request.method == 'POST':
        url_param_value = request.args.get('param_key')
        remote_ip = request.remote_addr
        logger.info(
            f"------------------------------------------------<SEP>------------------------------------------------\nA request from IP: {remote_ip} at {datetime.now()} with URL parameter: {url_param_value}")
        try:
            input_question = request.form['input_question'].encode('utf-8').decode('utf-8', 'ignore').strip().replace(
                "\n", " ").replace("\r", " ").strip()
            if not isinstance(input_question, str) or len(input_question) == 0:
                return render_template('demo.html', alert="PLEASE INPUT A VALID QUESTION!", info=info, display="none")
            model.generate_demo(input_question)
            info = Info(
                question=model.question,
                prediction=model.prediction,
                prediction_postprocessed=model.prediction_postprocessed,
                info_dict=model.info_dict,
                entity_mapping=model.entity_mapping,
                string_mapping=model.string_mapping,
                number=model.number,
                templates=model.templates,
                templates_postprocessed=model.templates_postprocessed,
                final_query=model.final_query,
                final_query_template=model.final_query_template,
                final_query_template_postprocessed=model.final_query_template_postprocessed,
                final_answer=model.final_answer,
                final_answer_dict=model.final_answer_dict,
                current_number=model.current_number,
                current_string=model.current_string,
                current_entity=model.current_entity,
                updated_entity_mapping=model.updated_entity_mapping,
                updated_string_mapping=model.updated_string_mapping,
            )
        except Exception as e:
            logger.info(f"ERROR WHEN invoking index(): {e}")
            traceback.print_exc()
            return render_template('demo.html', alert="SORRY THAT AN ERROR OCCURRED, PLEASE TRY AGAIN!", info=info,
                                   display="none")
        else:
            response = make_response(render_template('demo.html', alert=None, info=info, curse_to="predicted_sparql",display="block"))
            if 'info_json_' + remote_ip in session.keys():
                del session['info_json_' + remote_ip]
            logger.info(f"Setting session['info_json_{remote_ip}'] ignoring answers: \n{json.dumps(info.dict())}")
            info.final_answer_dict = {}
            info.final_answer = []
            session['info_json_' + remote_ip] = json.dumps(info.dict())
            return response
    else:
        return render_template('demo.html', alert=None, info=info, display="none")

@app.route('/update_template', methods=['POST'])
def update_template():
    remote_ip = request.remote_addr
    logger.info(f"******************************************\nContinue to invoke update_template() from IP: {remote_ip}")
    try:
        info_dict = json.loads(session.get('info_json_' + remote_ip, '{}'))
        info = Info(**info_dict)
        model.initialize(**info_dict)
        model.update_template(request.json["chosen_template"])
        info.current_entity = model.current_entity
        info.final_query_template_postprocessed = model.final_query_template_postprocessed
        info.final_query_template = model.final_query_template
        info.final_query = model.final_query
    except Exception as e:
        logger.info(f"ERROR WHEN INVOKING update_template(): {e}")
        traceback.print_exc()
        return make_response(jsonify({'updated_query': None}))
    else:
        response = make_response(jsonify({'updated_query': model.final_query}))
        if 'info_json_' + remote_ip in session.keys():
            del session['info_json_' + remote_ip]
        session['info_json_' + remote_ip] = json.dumps(info.dict())
        logger.info(f"Setting session['info_json_{remote_ip}']: \n{session.get('info_json_' + remote_ip, '{}')}")
        return response


@app.route('/update_number', methods=['POST'])
def update_number():
    remote_ip = request.remote_addr
    logger.info(f"******************************************\nContinue to invoke update_number() from IP: {remote_ip}")
    try:
        info_dict = json.loads(session.get('info_json_' + remote_ip, '{}'))
        info = Info(**info_dict)
        model.initialize(**info_dict)

        model.update_number(int(float(request.json["new_number_value"])))
        info.current_number = model.current_number
        info.current_entity = model.current_entity
        info.final_query = model.final_query
    except Exception as e:
        logger.info(f"ERROR WHEN INVOKING update_number(): {e}")
        traceback.print_exc()
        return make_response(jsonify({'updated_query': None}))
    else:
        if 'info_json_' + remote_ip in session.keys():
            del session['info_json_' + remote_ip]
        response = make_response(jsonify({'updated_query': model.final_query, 'updated_number': model.current_number}))
        session['info_json_' + remote_ip] = json.dumps(info.dict())
        logger.info(f"Setting session['info_json_{remote_ip}']: \n{session.get('info_json_' + remote_ip, '{}')}")
        return response

@app.route('/update_string', methods=['POST'])
def update_string():
    remote_ip = request.remote_addr
    logger.info(f"******************************************\nContinue to invoke update_string() from IP: {remote_ip}")
    try:
        info_dict = json.loads(session.get('info_json_' + remote_ip, '{}'))

        info = Info(**info_dict)
        model.initialize(**info_dict)
        string_dict=model.updated_string_mapping
        string_dict[request.json["string_label"]]=[request.json["selectedValue"]]
        model.update_string(string_dict)

        info.updated_string_mapping=model.updated_string_mapping
        info.current_string=model.current_string
        info.current_entity = model.current_entity
        info.final_query = model.final_query
    except Exception as e:
        logger.info(f"ERROR WHEN INVOKING update_string(): {e}")
        traceback.print_exc()
        return make_response(jsonify({'updated_query': None}))
    else:
        if 'info_json_' + remote_ip in session.keys():
            del session['info_json_' + remote_ip]
        response = make_response(jsonify({'updated_query': model.final_query}))
        session['info_json_' + remote_ip] = json.dumps(info.dict())
        logger.info(f"Setting session['info_json_{remote_ip}']: \n{session.get('info_json_' + remote_ip, '{}')}")

        return response

@app.route('/update_entity', methods=['POST'])
def update_entity():
    remote_ip = request.remote_addr
    logger.info(f"******************************************\nContinue to invoke update_entity() from IP: {remote_ip}")
    try:
        info_dict = json.loads(session.get('info_json_' + remote_ip, '{}'))
        info = Info(**info_dict)
        model.initialize(**info_dict)
        entity_dict=model.updated_entity_mapping
        entity_dict[request.json["topic_label"]]=[request.json["topic_value"]]
        model.update_entity(entity_dict)
        info.updated_entity_mapping = model.updated_entity_mapping
        info.current_entity = model.current_entity
        info.final_query = model.final_query
    except Exception as e:
        logger.info(f"ERROR WHEN INVOKING update_entity(): {e}")
        traceback.print_exc()
        return make_response(jsonify({'updated_query': None}))
    else:
        if 'info_json_' + remote_ip in session.keys():
            del session['info_json_' + remote_ip]
        response = make_response(jsonify({'updated_query': model.final_query}))
        session['info_json_' + remote_ip] = json.dumps(info.dict())
        logger.info(f"Setting session['info_json_{remote_ip}']: \n{session.get('info_json_' + remote_ip, '{}')}")
        return response

@app.route('/run_query', methods=['POST'])
def run_query():
    if request.method == 'POST':
        remote_ip = request.remote_addr
        logger.info(f"******************************************\nContinue to invoke run_query() from IP: {remote_ip}")
        try:
            info_dict = json.loads(session.get('info_json_' + remote_ip, '{}'))
            info = Info(info_dict["question"], info_dict["prediction"], info_dict["prediction_postprocessed"],
                        info_dict["info_dict"],
                        info_dict["entity_mapping"], info_dict["string_mapping"],
                        info_dict["number"], info_dict["templates"], info_dict["templates_postprocessed"],
                        info_dict["final_query"],
                        info_dict["final_query_template"],
                        info_dict["final_query_template_postprocessed"], info_dict["final_answer"], info_dict["final_answer_dict"],
                        info_dict["current_number"],
                        info_dict["current_string"],
                        info_dict["current_entity"],
                        info_dict["updated_entity_mapping"],
                        info_dict["updated_string_mapping"],
                        )
            model.initialize(**info_dict)
            model.final_query = request.json['input_sparql_value'].encode('utf-8').decode('utf-8', 'ignore').strip().replace(
                "\n", " ").replace("\r", " ").replace("  ", " ").strip()
            if not isinstance(model.final_query, str) or len(model.final_query) == 0:
                return make_response(jsonify({'updated_query': request.json['input_sparql_value'],'final_answer':None,'final_answer_dict':None}))
            model.single_query()
            info.final_query = model.final_query
            info.final_answer = model.final_answer
            info.final_answer_dict = model.final_answer_dict
        except Exception as e:
            logger.info(f"ERROR WHEN invoking run_query(): {e}")
            traceback.print_exc()
            return make_response(jsonify({'updated_query': None,'final_answer': None,'final_answer_dict':None}))
        else:
            if 'info_json_' + remote_ip in session.keys():
                del session['info_json_' + remote_ip]
            response = make_response(jsonify({'updated_query': request.json['input_sparql_value'],'final_answer':model.final_answer,'final_answer_dict':model.final_answer_dict}))

            logger.info(f"Setting session['info_json_{remote_ip}'] ignoring answers: \n{json.dumps(info.dict())}")
            info.final_answer_dict = {}
            info.final_answer = []
            session['info_json_' + remote_ip] = json.dumps(info.dict())
            return response

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    # app.run('0.0.0.0', port=8087, debug=True)
    app.run(debug=True)

import json
import random
import torch
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
import pandas as pd
from agatha.construct.semrep_handler import SemRepHandler
from agatha.ml.hypothesis_predictor import HypothesisPredictor
from agatha.util.matrix_lookup import np_emb_lookup_table, np_graph


class SemRepProcessor:
    def __init__(self, nlm_soft_folder, sr_temp_folder, sr_replace_utf8_path, sr_binary_path):
        self.sr_h = SemRepHandler(
            nlm_soft_path=nlm_soft_folder,
            temp_folder=sr_temp_folder,
            replace_utf8_path=sr_replace_utf8_path,
        )
        self.sr_h.sr_binary_path = Path(sr_binary_path)

    def process_text(self, text):
        cleaned_text = text.replace('\n', '').replace('*', '')
        start_time = time.time()
        expl_sr_out = self.sr_h.ProcessList_parallel([cleaned_text])
        end_time = time.time()
        # print(f"Time taken for SemRep processing: {end_time - start_time:.4f} seconds")
        return expl_sr_out


class ModelManager:
    def __init__(self, model_path, embedding_path, entity_db, graph_db, load_emb_into_ram=False):
        self.model_path = model_path
        self.embedding_path = embedding_path
        self.entity_db = entity_db
        self.graph_db = graph_db
        self.load_emb_into_ram = load_emb_into_ram
        self.model = self.load_model()

    def load_model(self):
        model = HypothesisPredictor.load_from_checkpoint(self.model_path)
        nodelbl_to_int_id_dict = json.load(open(self.entity_db))
        emb_np = np_emb_lookup_table(
            nodelbl_to_int_id_dict,
            self.embedding_path,
            memmap=not self.load_emb_into_ram,
        )
        graph_csr = sparse.load_npz(self.graph_db)
        graph_db = np_graph(nodelbl_to_int_id_dict, graph_csr)
        model.embeddings = emb_np
        model.graph = graph_db
        return model.eval()


class NegativeSampler:
    def __init__(self, st_nodelbl_dict):
        self.st_nodelbl_dict = st_nodelbl_dict

    def generate_negatives(self, pos_pair, sample_rate=5):
        s, o = pos_pair
        if s[0] != 'm':
            s = f'm:{s.lower()}'
        if o[0] != 'm':
            o = f'm:{o.lower()}'

        if o in self.st_nodelbl_dict:
            o_st = self.st_nodelbl_dict[o]
            available_population_size = len(self.st_nodelbl_dict[o_st])
            sample_size = min(sample_rate, available_population_size)
            o_sample_list = random.sample(self.st_nodelbl_dict[o_st], sample_size) if sample_size > 0 else []
        else:
            o_sample_list = []

        return [(s, o_neg) for o_neg in o_sample_list]


class PairEvaluator:
    def __init__(self, model, negative_sampler):
        self.model = model
        self.negative_sampler = negative_sampler


    def eval_pair(self, pair, sample_rate=50, n_runs=5):
        pair_negs = self.negative_sampler.generate_negatives(pair, sample_rate=sample_rate)
        agatha_queries = [pair] + pair_negs
        pair_labels = [1] + [0] * len(pair_negs)
    
        all_scores = []
        for _ in range(n_runs):
            scores = self.model.predict_from_terms(agatha_queries)
            all_scores.append(scores)
    
        # Average the scores across runs
        import numpy as np
        avg_scores = np.mean(all_scores, axis=0)
    
        res_list = sorted(
            zip(avg_scores, pair_labels),
            key=lambda x: x[0],
            reverse=True
        )
    
        rank = next((i for i, (_, lbl) in enumerate(res_list) if lbl == 1), None)
        return {
            'scores': res_list,
            'pos_rank': rank + 1 if rank is not None else None
        }


    # def eval_pair(self, pair, sample_rate=10):
    #     pair_negs = self.negative_sampler.generate_negatives(pair, sample_rate=sample_rate)
    #     agatha_queries = [pair] + pair_negs
    #     pair_labels = [1] + [0] * len(pair_negs)

    #     start_time = time.time()
    #     scores = self.model.predict_from_terms(agatha_queries)
    #     end_time = time.time()
    #     # print(f"Time taken for predict_from_terms from Agatha: {end_time - start_time:.4f} seconds")

    #     res_list = sorted(
    #         zip(scores, pair_labels),
    #         key=lambda x: x[0],
    #         reverse=True
    #     )

    #     rank = next((i for i, (_, lbl) in enumerate(res_list) if lbl == 1), None)
    #     return {
    #         'scores': res_list,
    #         'pos_rank': rank + 1 if rank is not None else None
    #     }


class AgathaEvaluator:
    def __init__(self, model_path, embedding_path, entity_db, graph_db, load_emb_into_ram=False):
        # Paths and configurations
        nlm_soft_folder = '/work/acslab/users/svalipou/SemRep'
        sr_temp_folder = '/work/acslab/users/svalipou/semrep_temp_jan_8_num_predicat_zero'
        sr_replace_utf8_path = '/work/acslab/users/svalipou/SemRep/replace_utf8.jar'
        sr_binary_path = '/work/acslab/users/svalipou/SemRep/public_semrep/bin/semrep.v1.9_2021AB'
        st_nodelbl_dict_path = '/work/acslab/shared/Agatha_shared/2022_11_22/2021_11_22_semtypes_to_nodelbl_and_vv_dict.json'

        # Initialize SemRepProcessor
        self.semrep_processor = SemRepProcessor(
            nlm_soft_folder,
            sr_temp_folder,
            sr_replace_utf8_path,
            sr_binary_path
        )
        with open(st_nodelbl_dict_path, 'r') as f:
            self.st_nodelbl_dict = json.load(f)
        self.negative_sampler = NegativeSampler(self.st_nodelbl_dict)
        self.model_manager = ModelManager(
            model_path,
            embedding_path,
            entity_db,
            graph_db,
            load_emb_into_ram
        )
        self.model = self.model_manager.model
        self.pair_evaluator = PairEvaluator(self.model, self.negative_sampler)
        self.pred_cui_pairs_set = self.extract_pred_cui_pairs()

    def extract_pred_cui_pairs(self):
        nodelist = self.model.graph.keys()
        pred_pairs_list = [p for p in tqdm(nodelist) if p[0] == 'p']
        pred_cui_pairs_set = set()

        for p in tqdm(pred_pairs_list):
            p_split = p.upper().split(':')
            s = p_split[1]
            o = p_split[-1]
            pred_cui_pairs_set.add(tuple(sorted([s, o])))

        return pred_cui_pairs_set

    def evaluate_llm_resp(self, llm_resp_str):
        expl_sr_out = self.semrep_processor.process_text(llm_resp_str)
        expl_parts_list = []
        incorrect_predicates_per_explanation = []

        for s_id, sent_data in expl_sr_out.items():
            sent_text = sent_data['sent_text']
            status = 'ok'
            bad_predicates = []
            good_predicates = []
            agatha_score_path = []
            # Get all predicates from relations
            all_predicates = [(rel['subj_text'], rel['verb'], rel['obj_text']) 
                             for rel in sent_data['relations']]

            for rel in sent_data['relations']:
                rel_subj = rel['subj_text']
                v = rel['verb']
                rel_obj = rel['obj_text']

                pred = [rel_subj, rel_obj]
                rel_pair = tuple(sorted([rel['subj_id'], rel['obj_id']]))

                if rel_pair in self.pred_cui_pairs_set:
                    good_predicates.append(pred)
                else:
                    subj_id = f"m:{rel['subj_id'].lower()}"
                    obj_id = f"m:{rel['obj_id'].lower()}"
                    if subj_id in self.model.graph and obj_id in self.model.graph:
                        eval_result = self.pair_evaluator.eval_pair([subj_id, obj_id], sample_rate=50)
                        agatha_score = eval_result['scores']
                        agatha_rank = eval_result['pos_rank']
                        pred.append(agatha_rank)
                        agatha_score_path.append(agatha_score)

                        # if agatha_rank > 1:
                        if agatha_rank > 5:
                            status = 'REWORK'
                            bad_predicates.append(pred)
                        else:
                            good_predicates.append(pred)

            expl_parts_list.append({
                'sent_text': sent_text,
                'status': status,
                'bad_predicates': bad_predicates,
                'good_predicates': good_predicates,
                'score': agatha_score_path,
                'all_predicates': all_predicates
            })
            incorrect_predicates_per_explanation.append(bad_predicates)

        return expl_parts_list, incorrect_predicates_per_explanation

    
    def diagnose_llm_response(self, llm_response_str, sample_rate=5):
        diagnostics = []
    
        expl_sr_out = self.semrep_processor.process_text(llm_response_str)
    
        for s_id, sent_data in expl_sr_out.items():
            sent_diag = {
                'sentence_id': s_id,
                'sentence_text': sent_data['sent_text'],
                'predicates': []
            }
    
            for rel in sent_data['relations']:
                subj_text = rel['subj_text']
                obj_text = rel['obj_text']
                verb = rel['verb']
                subj_id = f"m:{rel['subj_id'].lower()}"
                obj_id = f"m:{rel['obj_id'].lower()}"
                subj_type = rel.get('subj_semtypes', ['UNKNOWN'])
                obj_type = rel.get('obj_semtypes', ['UNKNOWN'])
    
                rel_pair = tuple(sorted([rel['subj_id'], rel['obj_id']]))
                pred_pair = [subj_id, obj_id]
    
                pred_info = {
                    'subj_text': subj_text,
                    'obj_text': obj_text,
                    'verb': verb,
                    'subj_type': subj_type,
                    'obj_type': obj_type,
                    'subj_id': subj_id,
                    'obj_id': obj_id,
                    'known_predicate': rel_pair in self.pred_cui_pairs_set,
                    'agatha_score': None,
                    'agatha_rank': None,
                    'label': None,
                    'negative_samples': []
                }
    
                if pred_info['known_predicate']:
                    pred_info['label'] = 'GOOD'
                elif subj_id in self.model.graph and obj_id in self.model.graph:
                    eval_result = self.pair_evaluator.eval_pair(pred_pair, sample_rate=sample_rate)
                    pred_info['agatha_score'] = eval_result['scores']
                    pred_info['agatha_rank'] = eval_result['pos_rank']
                    pred_info['label'] = 'BAD' if eval_result['pos_rank'] > 5 else 'GOOD'
                    pred_info['negative_samples'] = self.pair_evaluator.negative_sampler.generate_negatives(pred_pair, sample_rate=sample_rate)
                else:
                    pred_info['label'] = 'NOT_IN_GRAPH'
    
                sent_diag['predicates'].append(pred_info)
    
            diagnostics.append(sent_diag)
    
        return diagnostics

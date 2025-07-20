import random
import logging
from pathlib import Path
import time
from datetime import datetime
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import numpy as np

try:
    from llm_explainer.llm_explainer import vllm_get_response
    from hgcr_util.lazy_json_kv_loader import LazyJsonlAbstractLoader
    from agatha_evaluator import AgathaEvaluator
    from hgcr_util.emb_lookup.medcpt_emb_lookup import MedCPTNumpyEmbeddings
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    raise

# Define paths
WORKING_FOLDER = '/work/acslab/shared/Agatha_data/models_data/2021_11_22_reg_np_dl'
PATHS = {
    'agatha_sent_db': Path('/work/acslab/shared/Agatha_shared/pmid_to_sent_id_w_text_kv_jsonl_lazy_chunks'),
    'model': f'{WORKING_FOLDER}/model_reg_epoch_7.cpkt',
    'embedding': f'{WORKING_FOLDER}/2021_11_22_predicate_embeddings_512.npy',
    'entity_db': f'{WORKING_FOLDER}/21_11_22_predicate_entities_nodelbl_to_int_id_dict.json',
    'graph_db': f'{WORKING_FOLDER}/21_11_22_predicate_graph_csr.npz',
    'medcpt_emb': '/work/acslab/shared/MedCPT/medcpt_emb/'
}

# Initialize MedCPT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer_query = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
model_query = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)

# Initialize the precomputed embeddings object from HGCR
medcpt_emb_obj = MedCPTNumpyEmbeddings(
    medcpt_fpath='/work/acslab/shared/MedCPT/medcpt_emb/',
)

def encode_queries_medcpt(queries):
    with torch.no_grad():
        encoded = tokenizer_query(
            queries,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=64,
        ).to(device)
        embeddings = model_query(**encoded).last_hidden_state[:, 0, :]
    return embeddings

class LLMContextProcessor:
    def __init__(self, sents_db, medcpt_emb_obj, log_dir="./logs"):
        self.logger = self.setup_logger(log_dir)
        self.sents_db = sents_db
        self.medcpt_emb_obj = medcpt_emb_obj
        
        # Add file existence checks with detailed logging
        for key, path in PATHS.items():
            file_path = Path(path)
            try:
                # List directory contents to help debug
                parent_dir = file_path.parent
                self.logger.info(f"Contents of {parent_dir}:")
                for item in parent_dir.glob('*'):
                    self.logger.info(f"  {item.name}")
                
                if not file_path.exists():
                    error_msg = f"Required file not found: {key} at {path}"
                    self.logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                else:
                    self.logger.info(f"Found {key} at {path}")
                
                # Additional check for read permissions
                if not os.access(file_path, os.R_OK):
                    error_msg = f"No read permission for file: {key} at {path}"
                    self.logger.error(error_msg)
                    raise PermissionError(error_msg)
                
            except Exception as e:
                self.logger.error(f"Error checking {key} at {path}: {str(e)}")
                raise
        
        try:
            self.agatha_evaluator = AgathaEvaluator(
                model_path=PATHS['model'],
                embedding_path=PATHS['embedding'],
                entity_db=PATHS['entity_db'],
                graph_db=PATHS['graph_db'],
                load_emb_into_ram=False
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize AgathaEvaluator: {str(e)}")
            raise

    @staticmethod
    def setup_logger(log_dir):
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir_path / f"llm_context_log_{timestamp}.txt"

        logger = logging.getLogger("LLM_Context_Logger")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger
###########################################################################################################
    def process_df_with_llm(self, df, max_iterations, k, output_path='iteration_logs.pkl'):
        # Initialize logger
        logger = self.logger

        # INITIALIZATION lists to store results for each row
        iteration_results_per_row = []
        responses_per_row = []
        response_times_per_row = []
        evaluation_times_per_row = []
        total_evaluation_times_per_row = []
        prompts_per_iteration = []
        pmid_contexts_per_iteration = []
        pmid_changes_per_iteration = []
        termination_reasons = []
        iterations_completed = []
        final_explanations = []  # New list to store final explanations

        # STEP1: Taking column from DF 
        for idx, row in df.iterrows():  # idx will be the actual index (35, 58, etc.)
            source_cui = row["subj_name"]
            target_cui = row["obj_name"]
            # path_context_pmids = row["context_pmids"]
            path_context_pmids = row["final_context_pmids"]
            selected_pmids = {edge: pmids[:k] for edge, pmids in path_context_pmids.items()}
            used_pmids = {edge: set(pmids[:k]) for edge, pmids in path_context_pmids.items()}
            iteration_logs = []

            # Initialize these variables at the start of each row's processing
            iteration_pmid_changes = []
            iteration_pmid_scores = []

            logger.info("="*80)
            logger.info(f"Processing DataFrame Row {idx}: Source={source_cui}, Target={target_cui}")
            logger.info("="*80)

            # Initialize row-specific tracking lists
            row_responses = []
            row_response_times = []
            row_evaluation_times = []
            row_prompts = []
            row_pmid_contexts = []
            row_pmid_changes = []
            row_pmid_scores = []

            termination_reason = None

            # STEP2: for each iteration start preparing fixed prompt + context
            for iteration in range(max_iterations):
                # Store current iteration context
                current_iteration_context = selected_pmids.copy()

                # Log the context PMIDs being used for this iteration
                logger.info("\n" + "="*80)
                logger.info(f"ROW {idx}, ITERATION {iteration+1} - CONTEXT PMIDs USED:")
                for edge, pmids in current_iteration_context.items():
                    logger.info(f"Edge {edge}: {pmids}")
                logger.info("="*80)

                # Prepare prompt and get response
                prompt = self._prepare_prompt(source_cui, target_cui, current_iteration_context)
                logger.info("\n" + "="*80)
                logger.info(f"ROW {idx}, ITERATION {iteration+1} - PROMPT:")
                logger.info(prompt)
                logger.info("="*80)

                # Get LLM response and log it
                start_time = time.time()
                # STEP3: send to llm to receive response
                response = vllm_get_response(prompt)
                response_time = time.time() - start_time
                
                logger.info("\n" + "="*80)
                logger.info(f"ROW {idx}, ITERATION {iteration+1} - LLM RESPONSE:")
                logger.info(response)
                logger.info(f"Response time: {response_time:.2f} seconds")
                logger.info("="*80)

                # STEP4: Evaluate response from llm in evaluate_llm_res via Agatha
                eval_start_time = time.time()
                evaluation_result, _ = self.agatha_evaluator.evaluate_llm_resp(response)
                evaluation_time = time.time() - eval_start_time

                logger.info("\n" + "="*80)
                logger.info(f"ROW {idx}, ITERATION {iteration+1} - EVALUATION RESULTS:")
                for i, explanation in enumerate(evaluation_result, 1):
                    logger.info(f"\nSentence {i}:")
                    logger.info(f"Text: {explanation['sent_text']}")
                    logger.info(f"Status: {explanation['status']}")
                    if explanation['status'] == 'REWORK':
                        logger.info(f"Bad Predicates: {explanation.get('bad_predicates', [])}")
                logger.info(f"\nEvaluation time: {evaluation_time:.2f} seconds")
                logger.info("="*80)

                # # INITIALIZATION list for result out of evaluate_llm_res
                all_bad_predicates = []
                all_predicates = []
                total_number_of_predicates = 0
                rework_sentences = []

                '''
               It checks for REWORK status sentences and collects them in rework_sentences list
               If no REWORK sentences are found (rework_sentences is empty), it:
                   Sets termination reason to "No REWORK Sentences Found"
                   Breaks the iteration loop
                   Moves to the next row in the DataFrame
               If one or more REWORK sentences are found, it:
                   Processes each sentence individually in the loop
                   Finds closest PMID for each sentence
                   Makes necessary replacements for each sentence for context of pmids
                   Stores the changes and scores for each sentence 
                '''

                # Caliberate result to log result out of evaluator_llm_res
                for explanation in evaluation_result:
                    if explanation['status'] == 'REWORK':
                        rework_sentences.append(explanation['sent_text'])
                        all_bad_predicates.extend(explanation.get('bad_predicates', []))
                    
                    # Update how we collect predicates
                    all_predicates.extend(explanation.get('all_predicates', []))
                    total_number_of_predicates = len(all_predicates)

                # Store iteration results
                iteration_log = {
                    'iteration': iteration + 1,
                    'response': response,
                    'response_time': response_time,
                    'evaluation_time': evaluation_time,
                    'rework_sentences': rework_sentences,
                    'bad_predicates': all_bad_predicates,
                    'good_predicates': explanation.get('good_predicates', []),  # Add good predicates
                    'all_predicates': all_predicates,
                    'total_number_of_predicates': total_number_of_predicates,
                    'prompt': prompt,
                    'context_used': current_iteration_context.copy(),
                    'pmid_changes': iteration_pmid_changes,
                    'pmid_scores': iteration_pmid_scores
                }
                iteration_logs.append(iteration_log)

                # Append iteration-specific data
                row_responses.append(response)
                row_response_times.append(response_time)
                row_evaluation_times.append(evaluation_time)
                row_prompts.append(prompt)
                row_pmid_contexts.append(current_iteration_context.copy())
                row_pmid_changes.extend(iteration_pmid_changes)
                row_pmid_scores.extend(iteration_pmid_scores)

                # Check termination condition
                if not rework_sentences:
                    termination_reason = "No REWORK Sentences Found"
                    logger.info("No REWORK sentences found, terminating iterations.")
                    break

                # Process REWORK sentences
                iteration_pmid_changes = []
                iteration_pmid_scores = []

                for sentence in rework_sentences:
                    edge, closest_pmid, pmid_scores = self._find_closest_pmid(
                        sentence,
                        current_iteration_context,
                        idx, iteration,
                        source_cui, target_cui,
                        len(df)
                    )
                    
                    pmid_change = self._replace_pmid(
                        edge,
                        selected_pmids,
                        path_context_pmids,
                        used_pmids,
                        pmid_scores
                    )
                    
                    if pmid_change:
                        iteration_pmid_changes.append(pmid_change)
                        iteration_pmid_scores.append({
                            'sentence': sentence,
                            'scores': pmid_scores,
                            'selected_change': pmid_change
                        })

            if not termination_reason:
                termination_reason = "Max Iterations Reached"

            # Store row results
            iteration_results_per_row.append(iteration_logs)
            responses_per_row.append(row_responses)
            response_times_per_row.append(row_response_times)
            evaluation_times_per_row.append(row_evaluation_times)
            total_evaluation_times_per_row.append(sum(row_evaluation_times))
            prompts_per_iteration.append(row_prompts)
            pmid_contexts_per_iteration.append(row_pmid_contexts)
            pmid_changes_per_iteration.append(row_pmid_changes)
            termination_reasons.append(termination_reason)
            iterations_completed.append(iteration + 1)
            final_explanations.append(row_responses[-1] if row_responses else None)  # Add the last response

        # Update DataFrame with results
        df['iteration_results'] = iteration_results_per_row
        df['respond_llm'] = responses_per_row
        df['respond_time_llm'] = response_times_per_row
        df['evaluation_time_llm'] = evaluation_times_per_row
        df['total_evaluation_time_per_row'] = total_evaluation_times_per_row
        df['prompts_per_iteration'] = prompts_per_iteration
        df['pmid_contexts_per_iteration'] = pmid_contexts_per_iteration
        df['pmid_changes_per_iteration'] = pmid_changes_per_iteration
        df['termination_reason'] = termination_reasons
        df['iterations_completed'] = iterations_completed
        df['final_explanation'] = [responses[-1] for responses in df['respond_llm']]  # Get last response from respond_llm column

        # Save results
        df.to_pickle(output_path)
        logger.info(f"DataFrame saved to {output_path}")

        return df

###########################################################################################################    
    def _find_closest_pmid(self, sentence, iteration_context, idx, iteration, source_cui, target_cui, total_rows):
        """
        Find the closest PMID and its edge from the context used to generate the current response.
        
        Args:
            sentence: The REWORK sentence to compare against
            iteration_context: Dict[Tuple[str, str], List[str]] - Context used for this iteration
                             where keys are edges (source_cui, target_cui) and values are PMID lists
        """
        query_embeddings = encode_queries_medcpt([sentence]).to(device)
        
        # Track PMIDs and their corresponding edges
        pmid_to_edge = {}
        pmid_list = []
        pmid_embeddings_list = []
        
        # Process PMIDs maintaining edge information
        for edge, pmids in iteration_context.items():
            for pmid in pmids:
                pmid_to_edge[pmid] = edge
                pmid_list.append(pmid)
                if pmid in medcpt_emb_obj:
                    pmid_embedding = medcpt_emb_obj[pmid]
                    if isinstance(pmid_embedding, np.ndarray):
                        pmid_embeddings_list.append(pmid_embedding)
                        self.logger.info(f"Edge {edge}: Processing PMID {pmid}")
                    else:
                        self.logger.warning(f"Edge {edge}: Invalid embedding for PMID {pmid}")
                else:
                    self.logger.warning(f"Edge {edge}: PMID {pmid} not found in embeddings")
        
        if not pmid_embeddings_list:
            raise ValueError("No valid embeddings found for the provided PMIDs")
        
        pmid_embeddings = torch.tensor(np.vstack(pmid_embeddings_list), dtype=torch.float32).to(device)
        
        # Perform semantic search
        search_results = util.semantic_search(
            query_embeddings, pmid_embeddings, top_k=len(pmid_list), score_function=util.cos_sim
        )
        
        # Create scores dictionary with edge information
        pmid_scores = {}
        for result in search_results[0]:
            pmid = pmid_list[result['corpus_id']]
            edge = pmid_to_edge[pmid]
            pmid_scores[pmid] = {
                'score': result['score'],
                'edge': edge
            }
        
        # Log results
        self.logger.info(f"Iteration {iteration} - Analyzing sentence: {sentence}")
        for pmid, data in pmid_scores.items():
            self.logger.info(f"Edge {data['edge']}: PMID {pmid} - Score {data['score']:.4f}")
        
        # Get the closest PMID and its edge
        closest_pmid = pmid_list[search_results[0][0]['corpus_id']]
        closest_edge = pmid_to_edge[closest_pmid]
        
        return closest_edge, closest_pmid, pmid_scores

###########################################################################################################    
    def _replace_pmid(self, edge, selected_pmids, path_context_pmids, used_pmids, pmid_scores):
        """
        Replace a PMID for a specific edge in the context.
        
        Args:
            edge: Tuple[str, str] - The edge (source_cui, target_cui) to modify
            selected_pmids: Dict[Tuple[str, str], List[str]] - Current context
            path_context_pmids: Dict[Tuple[str, str], List[str]] - All available PMIDs per edge
            used_pmids: Dict[Tuple[str, str], Set[str]] - Already used PMIDs per edge
            pmid_scores: Dict[str, Dict] - Scores and edge info for PMIDs
        """
        # Find the PMID with highest similarity score for this edge
        edge_pmids = {pmid: data['score'] 
                     for pmid, data in pmid_scores.items() 
                     if data['edge'] == edge and pmid in selected_pmids[edge]}
        
        if not edge_pmids:
            self.logger.warning(f"No scored PMIDs found for edge {edge}")
            return None
            
        # Select PMID with highest score (most similar to problematic sentence) to drop
        pmid_to_drop = max(edge_pmids.items(), key=lambda x: x[1])[0]
        
        remaining_pmids = [pmid for pmid in selected_pmids[edge] if pmid != pmid_to_drop]
        
        # Try to find an unused PMID for this edge
        available_pmids = [pmid for pmid in path_context_pmids[edge] 
                          if pmid not in used_pmids[edge]]
        
        if available_pmids:
            new_pmid = available_pmids[0]
            selected_pmids[edge] = remaining_pmids + [new_pmid]
            used_pmids[edge].add(new_pmid)
            
            self.logger.info(f"Edge {edge}: Replaced PMID {pmid_to_drop} (score: {edge_pmids[pmid_to_drop]:.4f}) "
                           f"with new PMID {new_pmid}")
            
            return {
                "edge": edge,
                "dropped": pmid_to_drop,
                "added": new_pmid
            }
        else:
            self.logger.warning(f"Edge {edge}: No unused PMIDs available to replace {pmid_to_drop}")
            selected_pmids[edge] = remaining_pmids + [pmid_to_drop]
            return {
                "edge": edge,
                "dropped": pmid_to_drop,
                "added": None
            }

###########################################################################################################
    def _prepare_prompt(self, source, target, selected_pmids):
        """
        Generate a prompt combining a fixed template with abstracts.
    
        Args:
            source (str): The source entity for the explanation.
            target (str): The target entity for the explanation.
            selected_pmids (dict): A dictionary where keys are edges (tuples) and 
                values are lists of PMIDs associated with each edge.
    
        Returns:
            str: A formatted prompt including the source, target, and retrieved abstracts.
    
        Raises:
            RuntimeError: If no abstracts are found for the selected PMIDs or PMIDs are missing from the database.
        """
        abstracts = []
        for edge, pmids in selected_pmids.items():
            for pmid in pmids:
                try:
                    abstract = self.sents_db[pmid]
                    if abstract:
                        abstracts.append(abstract)
                    else:
                        self.logger.error(f"No abstract found for PMID: {pmid}")
                        raise RuntimeError(f"No abstract found for PMID: {pmid}")
                except KeyError:
                    self.logger.error(f"PMID {pmid} not found in the database.")
                    raise RuntimeError(f"PMID {pmid} not found in the database.")

        if not abstracts:
            self.logger.error("No abstracts retrieved for the selected PMIDs.")
            raise RuntimeError("No abstracts retrieved for the selected PMIDs.")

        prompt_template = (
            "Based on the following scientific abstracts, please describe how an indirect relationship "
            "between {source} and {target} might exist. Consider the key findings, underlying mechanisms, "
            "and any intermediate entities or processes mentioned in the abstracts. Your explanation "
            "should connect these elements to form a coherent narrative that illustrates the possible "
            "indirect linkage between {source} and {target}."
        )
        return prompt_template.format(source=source, target=target) + "\n\n" + "\n\n".join(abstracts)


# coding=utf-8

"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm
from konlpy.tag import Mecab

import torch
import random
from transformers import is_torch_available, PreTrainedTokenizerFast
from transformers.trainer_utils import get_last_checkpoint
from pororo import Pororo
logger = logging.getLogger(__name__)


from konlpy.tag import Mecab
from konlpy.tag import Kkma
from konlpy.tag import Hannanum
import re

mecab = Mecab()
kkma = Kkma()
hannanum = Hannanum()

tokenizer_pororo = Pororo(task="tokenization", lang="ko", model="mecab.bpe64k.ko")
def tokenize(text):
    # return text.split(" ")
    #return mecab.morphs(text)
    return tokenizer_pororo(text)

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# 'JC','JX','JKS','JKC','JKG','JKO','JKB','JKV','JKQ','EP','EF','EC','ETN','ETM'

def postprocess(ans):
    if mecab.pos(ans)[-1][-1] in ["JX", "JKB", "JKO", "JKS", "ETM", "VCP", "JC"]:
        ans = ans[:-len(mecab.pos(ans)[-1][0])]
    elif ans[-1] == "의":
        wsd_result = wsd(ans)[-1]
        if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의'):
            ans = ans[:-1]

    if ans == '있':
        ans = ''
    elif ans == '네':
        ans = ''
    elif ans == '개월':
        ans = ''
    elif ans == '해서':
        ans = ''
    elif ans == '이':
        ans = ''
    elif ans == '신':
        ans = ''
    elif ans == '명':
        ans = ''
    elif ans == ',':
        ans = ''
    elif ans == '‘':
        ans = ''
    elif ans == '*':
        ans = ''
    elif ans == '.':
        ans = ''
    elif ans == '것':
        ans = ''
    elif ans == '_':
        ans = ''
    elif ans[-2:] == '일자':
        ans = ans[:-1]
    elif ans[-2:] == '년에':
        ans = ans[:-1]
    elif ans[-2:] == '년간':
        ans = ans[:-1]
    elif ans[-2:] == '였다':
        ans = ans[:-2]
    elif ans[-2:] == '이다':
        ans = ans[:-2]
    elif ans[-2:] == '이며':
        ans = ans[:-2]
    elif ans[-2:] == '위해':
        ans = ''
    elif ans[-2:] == '난이':
        ans = ans[:-1]
    elif ans[-3:] == '년대에':
        ans = ans[:-1]
    elif ans[-3:] == '인돌이':
        ans = ans[:-1]
    elif ans[-3:] == '대기에':
        ans = ans[:-1]
    elif ans[-3:] == '찰사인':
        ans = ans[:-1]
    elif ans[-3:] == '일린을':
        ans = ans[:-1]
    elif ans[-3:] == '리토와':
        ans = ans[:-1]
    elif ans[-3:] == '3장이':
        ans = ans[:-1]
    elif ans[-3:] == '의적인':
        ans = ans[:-2]
    elif ans[-3:] == '즐리가':
        ans = ans[:-1]
    elif ans[-3:] == '늠선이':
        ans = ans[:-1]
    elif ans[-3:] == '악가인':
        ans = ans[:-1]
    elif ans[-3:] == '이라고':
        ans = ans[:-2]
    elif ans[-3:] == '합니다':
        ans = ''
    elif ans[-3:] == '정해져':
        ans = ''
    elif ans == '호엔원년(1036':
        ans += ')'
    elif ans == 'lea et al (1987':
        ans += ')'
    elif ans == '외무대신과 농상무대신':
        ans += '외무대신'
    return ans

def find_more_span_in_context(predicted, context):
    if predicted not in context:
        ans_tmp = ''
    else:
        ans_tmp = predicted

    if ans_tmp != '':
        ans_tmp = postprocess(ans_tmp)
    else:
        ans_tmp = ''

    if ans_tmp.count('(') != ans_tmp.count(')'):
        ans_tmp = ans_tmp.replace('(','')
        ans_tmp = ans_tmp.replace(')','')

    if ans_tmp == '':
        pass
    elif "'" + ans_tmp + "'" in context:
        ans_tmp = "'" + ans_tmp + "'"
       
    elif '"' + ans_tmp + '"' in context:
        ans_tmp = '"' + ans_tmp + '"'

    elif '(' + ans_tmp + ')' in context:
        ans_tmp = '(' + ans_tmp + ')' 
    
    elif '“' + ans_tmp + '”' in context:
        ans_tmp = '“' + ans_tmp + '”'

    elif '‘' + ans_tmp + '’' in context:
        ans_tmp = '‘' + ans_tmp + '’'

    elif '《' + ans_tmp + '》' in context:
        ans_tmp = '《' + ans_tmp + '》'

    elif '≪' + ans_tmp + '≫' in context:
        ans_tmp = '≪' + ans_tmp + '≫'

    elif '〈' + ans_tmp + '〉' in context:
        ans_tmp = '〈' + ans_tmp + '〉'
 
    elif '『' + ans_tmp + '』' in context:
        ans_tmp = '『' + ans_tmp + '』'

    elif '「' + ans_tmp + '」' in context:
        ans_tmp = '「' + ans_tmp + '」'

    elif '＜' + ans_tmp + '＞' in context:
        ans_tmp = '＜' + ans_tmp + '＞'

    elif '{' + ans_tmp + '}' in context:
        ans_tmp = '{' + ans_tmp + '}'

    elif '<' + ans_tmp + '>' in context:
        ans_tmp = '<' + ans_tmp + '>'

    elif '[' + ans_tmp + ']' in context:
        ans_tmp = '[' + ans_tmp + ']'


    try:
        if ans_tmp != '':
            p = re.compile(ans_tmp + "\([\sㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔァ-ヴー々〆〤一-龥]*\)")
            m = p.findall(context)
            if len(m) != 0:
                ans_tmp = m[0]
    except:
        pass

    try:
        if ans_tmp != '':
            p = re.compile(ans_tmp + "\s\([\sㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9ぁ-ゔァ-ヴー々〆〤一-龥]*\)")
            m = p.findall(context)

            if len(m) != 0:
                ans_tmp = m[0]
    except:
        pass

    return ans_tmp



def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.

    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).

            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # Let's loop over all the examples!

    for example_index, example in enumerate(tqdm(examples)):
        
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]
        
        
        min_null_prediction = None
        prelim_predictions = []
        
        # Looping through all the features associated to the current example.
        # 하나의 query 에 대해 query-passage dataset이 여러개로 만들어질수 있으므로(passage가 길어서)
        for i,feature_index in enumerate(feature_indices):
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            #"score": start_logits[start_index] + 0.1*end_logits[end_index] - 0.5*i,
                            #"score": start_logits[start_index] + end_logits[end_index] - 0.5*i,
                            ## inference 할때
                            #"score": (5+start_logits[start_index])*(5+end_logits[end_index])/example['retriever_ratio_score'] ,
                            "score": start_logits[start_index]+end_logits[end_index], 
                            
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        
        
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size//2]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            predicted_before = context[offsets[0]:offsets[1]]
            print('predicted_before',predicted_before)
            #print('context',context)
            pred["text"] = find_more_span_in_context(predicted_before, context)
            print('predicted_after',pred["text"] )
            

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        # scores = np.array([pred.pop("score") for pred in predictions])
        scores = np.array([pred["score"] for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
        

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            all_predictions[example["id"]] = predictions[i]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(
                score_diff
            )  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n")
        
        # inference 할때
        #last_preprocessing(prediction_file,output_dir)

    return all_predictions

def last_preprocessing(prediction_file,output_dir):
    mecab = Mecab()
    kkma = Kkma()
    hannanum = Hannanum()
    with open(prediction_file, encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    
    for k, v in json_data.items():
        if mecab.pos(v)[-1][-1] in {"JX", "JKB", "JKO", "JKS", "ETM", "VCP", "JC"}:
            json_data[k] = json_data[k][:-len(mecab.pos(v)[-1][0])]
        elif v[-1] == "의":
            # 지울 수 있는 `의` 인지 check
            if kkma.pos(v)[-1][-1] == "JKG" or mecab.pos(v)[-1][-1] == "NNG" or hannanum.pos(v)[-1][-1] == "J":
                json_data[k] = json_data[k][:-1]
    
    prediction_post_file = os.path.join(output_dir, "prediction_post.json")

    with open(prediction_post_file , "w", encoding="utf-8") as make_file:
        json.dump(json_data, make_file, indent="\t", ensure_ascii=False)


def check_no_error(training_args, data_args, tokenizer, datasets):
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length
import logging
import os
import sys
import pickle
import pandas as pd
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval
#from retrieval_copy import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    
    # add question type tags as special token
    tag=['[CITE]', '[HOW]', '[QUANTITY]', '[WHAT]', '[WHEN]', '[WHERE]', '[WHO]', '[WHY]']
    special_tokens_dict = {'additional_special_tokens': tag}
    tokenizer.add_special_tokens(special_tokens_dict)
    print(tokenizer)

    # add compound nouns to dictionary
    with open('/opt/ml/code/data/compounds_test.pickle', 'rb') as f:
        compounds = pickle.load(f)
    tokenizer.add_tokens(list(compounds))
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # add additional layers to the header of the model (must include classifier)
    # nodes = model.qa_outputs.in_features
    # layers = [nn.Linear(in_features=nodes, out_features=nodes),
    #           nn.Linear(in_features=nodes, out_features=2)]
    # model._modules['qa_outputs'] = nn.Sequential(*layers)
    
    model.resize_token_embeddings(len(tokenizer))

    # run passage retrieval if true
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(datasets, training_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(datasets, training_args):
    #### retreival process ####

    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="./data",
                                context_path="wikipedia_documents.json"
                                # context_path="all_wikipedia_documents.json"
                                )
    # sparse embedding retrieval
    # retriever.get_sparse_embedding()
    #df = retriever.retrieve(datasets['validation'])

    # bm25 retrieval
    # retriever.get_embedding_BM25()
    # df = retriever.retrieve_BM25(query_or_dataset=datasets['validation'], topk=10)

    # elastic search retrieval
    # retriever.get_elastic_search()
    df = retriever.retrieve_ES(query_or_dataset=datasets['validation'], topk=10)

    # faiss retrieval
    # df = retriever.retrieve_faiss(dataset['validation'])

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        f = Features({'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    elif training_args.do_eval: # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
        f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                                   'answer_start': Value(dtype='int32', id=None)},
                                          length=-1, id=None),
                      'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    # only for eval or predict
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    # check if there is an error
    last_checkpoint, max_seq_length = check_no_error(training_args, data_args, tokenizer, datasets)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
            #is_split_into_words=True
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]
    eval_tag = pd.read_csv("/opt/ml/code/data/test_tag.tsv", header=None)[0]

    def add_tag_eval(example, idx):
        example['question'] = eval_tag[idx] + example['question']
        return example

    eval_dataset = eval_dataset.map(add_tag_eval, with_indices=True)
    print(eval_dataset['question'][:2])

    # Validation Feature Creation
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data collator.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset= None,
        eval_dataset=eval_dataset,
        eval_examples=datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - will create predictions.json
    if training_args.do_predict:
        predictions = trainer.predict(test_dataset=eval_dataset,
                                        test_examples=datasets['validation'])

        # predictions.json is already saved when we call postprocess_qa_predictions(). so there is no need to further use predictions.
        print("No metric can be presented because there is no correct answer given. Job done!")

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

if __name__ == "__main__":
    main()

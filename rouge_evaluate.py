import os
import sys
import torch
import argparse
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "KoBART"))
from tqdm import tqdm
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
from rouge_metric import Rouge

def evaluate_rouge(model_path, test_path, n, max_len, min_len, num_beams, penalty):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = get_kobart_tokenizer()

    texts = list()
    hypothesis = list()
    references = list()

    with open(test_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            news, summary = line.split("\t")
            texts.append(news)
            references.append(summary)

    for text in tqdm(texts):
        text = text.replace('\n', '')
        input_ids = tokenizer.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model.generate(
            input_ids, 
            eos_token_id=1, 
            max_length=max_len, 
            min_length=min_len, 
            num_beams=num_beams,
            length_penalty=penalty
        )
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        hypothesis.append(output)
        #print(output)

    rouge = Rouge(metrics=["rouge-n", "rouge-l", "rouge-w"], max_n=n)
    score = rouge.get_scores(hypothesis, references)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='kobart model path')
    parser.add_argument('--test_data_path',
                        type=str,
                        required=True,
                        help='path of test data in form of tsv file')
    parser.add_argument('--rouge_n',
                        type=int,
                        default=2,
                        help='max n value of rouge-n')
    parser.add_argument('--max_len',
                        type=int,
                        default=1024,
                        help='max length of generated sequence')
    parser.add_argument('--min_len',
                        type=int,
                        default=10,
                        help='min length of generated sequence')
    parser.add_argument('--num_beams',
                        type=int,
                        default=5,
                        help='number of beams for beam search')
    parser.add_argument('--penalty',
                        type=float,
                        default=1.0,
                        help='larger penalty than 1.0 means shorter sequences')
    args = parser.parse_args()
    scores = evaluate_rouge(
        args.model_path,
        args.test_data_path,
        args.rouge_n,
        args.max_len,
        args.min_len,
        args.num_beams,
        args.penalty
    )
    print("Metrics\tF1\tPrecision\tRecall")
    for score in scores:
        print("{}\t{:.3f}\t{:.3f}\t{:.3f}".format(score, scores[score]["f"], scores[score]["p"], scores[score]["r"]))
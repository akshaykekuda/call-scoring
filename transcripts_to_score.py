import argparse, configparser
from pathlib import Path
from PrepareDf import prepare_inference_df
from DatasetClasses import InferenceCallDataSet
from DataLoader_fns import Collate
from torch.utils.data import DataLoader
import torch
import pandas as pd
from tqdm import tqdm

sub_score_categories = ['Cross Selling', 'Creates Incentive', 'Education', 'Processes', 'Product Knowledge', 'Greeting', 'Professionalism', 'Confidence',  'Retention',
                        'Documentation']

score_wts = [5, 2, 5, 5, 5, 3, 5, 2, 4, 3]


def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='getting socre.py')

    # General system running and configuration options
    parser.add_argument('--transcript_path', type=str, help='path to transcripts for scoring')
    parser.add_argument('--model_path', type=str, help="path to trained model")
    parser.add_argument('--out_path', type=str, default='scores.csv', help="path to output csv")
    parser.add_argument('--vocab_path', type=str, default='call_vocab', help='path to call_vocab file')
    parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--num_workers', type=int, default=1, help='device to use')

    args = parser.parse_args()
    return args


def get_scores(dataloader, model, scoring_criterion):
    id_arr = []
    text_arr = []
    attn_score_arr = []
    pred_arr = []
    raw_pred_arr = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs, scores, _ = model(batch)
            output = outputs[0].reshape(-1, len(scoring_criterion), 2)
            probs = torch.softmax(output, dim=-1)
            max_vals = torch.max(probs, dim=-1)
            raw_proba = probs[:, :, 1].tolist()
            pred = max_vals[1].tolist()
            raw_pred_arr.extend(raw_proba)
            pred_arr.extend(pred)
            id_arr.extend(batch['id'])
            text_arr.extend(batch['text'])
            attn_score_arr.extend(scores.numpy())
        raw_pred_df = pd.DataFrame(raw_pred_arr, columns=['RawProba ' + category for category in scoring_criterion])
        pred_df = pd.DataFrame(pred_arr, columns=scoring_criterion)
        df = pd.concat((pred_df, raw_pred_df), axis=1)
        df['id'] = id_arr
        df['text'] = text_arr
        df['scores'] = attn_score_arr
    return df


def scale_df(df, scoring_criteria):
    for i in range(len(scoring_criteria)):
        df[scoring_criteria[i]] = (df[scoring_criteria[i]] != 1) * score_wts[i]
    df['CombinedPercentileScore'] = (df.iloc[:, :len(scoring_criteria)].sum(axis=1)/39) * 100
    return df.round(2)

    pass


if __name__ == "__main__":
    print("Start of inference")
    args = _parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    paths = [str(x) for x in Path(args.transcript_path).glob("**/*.txt")]
    df = prepare_inference_df(args.transcript_path)
    call_dataset = InferenceCallDataSet(df)
    with open(args.vocab_path, 'rb') as f:
        vocab = torch.load(f)
    c = Collate(vocab, args.device)
    call_dataloader = DataLoader(call_dataset, batch_size=4, shuffle=True,
                                              num_workers=args.num_workers, collate_fn=c.collate)
    model = torch.load(args.model_path)

    df = get_scores(call_dataloader, model, scoring_criterion=sub_score_categories)
    df = scale_df(df, sub_score_categories)
    df.to_csv(args.out_path, index=False)







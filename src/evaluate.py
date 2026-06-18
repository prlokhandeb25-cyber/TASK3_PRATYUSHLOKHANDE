import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from model import Transformer
from train import Vocabulary
import sacrebleu
import math


def greedy_decode(model, src, max_len, src_vocab, tgt_vocab, device):
    model.eval()
    src = src.to(device)

    src_mask = model.make_src_mask(src, pad_idx=0)
    src_emb = model.pos_encoding(
        model.src_embedding(src) * math.sqrt(model.d_model)
    )
    encoder_output = model.encoder(src_emb, src_mask)

    tgt_ids = [2]  

    for _ in range(max_len):
        tgt = torch.tensor(tgt_ids).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt, pad_idx=0)
        tgt_emb = model.pos_encoding(
            model.tgt_embedding(tgt) * math.sqrt(model.d_model)
        )
        decoder_output = model.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        output = model.output_layer(decoder_output)


        logits = output[0, -1, :]
        logits[1] = -1e9 
        next_token = logits.argmax().item()
        tgt_ids.append(next_token)

        if next_token == 3: 
            break

    return tgt_ids[1:-1]  


def evaluate():
    print("Loading model...")
    checkpoint = torch.load("best_model.pt", map_location="cpu", weights_only=False)
    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]
    d_model   = checkpoint["d_model"]

    model = Transformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=d_model,
        h=8,
        d_ff=512,
        N=3,
        dropout=0.1
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    device = torch.device("cpu")

    # Load test data
    from datasets import load_dataset
    dataset  = load_dataset("bentrevett/multi30k")
    test_data = dataset["test"]

    hypotheses = []
    references = []

    print("Translating test set...")
    for i, example in enumerate(test_data):
        src_sentence = example["en"]
        ref_sentence = example["de"]

        src_ids = [2] + src_vocab.encode(src_sentence) + [3]
        src_tensor = torch.tensor(src_ids).unsqueeze(0)

        pred_ids = greedy_decode(model, src_tensor, max_len=50,
                                 src_vocab=src_vocab, tgt_vocab=tgt_vocab,
                                 device=device)

        pred_tokens = [tgt_vocab.idx2word.get(i, "<unk>") for i in pred_ids]
        pred_sentence = " ".join(pred_tokens)

        hypotheses.append(pred_sentence)
        references.append(ref_sentence)

        if i < 5:
            print(f"\nSRC : {src_sentence}")
            print(f"REF : {ref_sentence}")
            print(f"PRED: {pred_sentence}")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n{'='*50}")
    print(f"BLEU SCORE: {bleu.score:.2f}")
    print(f"(Paper base model: 27.3 | Our CPU model:{bleu.score:.2f})")
    print(f"{'='*50}")


if __name__ == "__main__":
    evaluate()
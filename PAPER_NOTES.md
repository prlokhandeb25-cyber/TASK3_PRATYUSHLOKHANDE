## Central Claim : 
The paper written by published by Google Brain, revolves around the idea that instead of the traditional RNN(Recurrent Neural Networks) and LSTMs ( Long Short Term Memory Networks) which used sequential data handling , we can use a 'Transformer' which only requires 'ATTENTION' which is a function which maps a query and a set of key value pairs to an output . Here the query,key,value,and output all are vectors . The use of a transformer helps models to process data more efficiently and supports parallelism . Main problems with RNN,LSTMs was sequential processing which made training slow and it struggled with long term dependecies i.e. as it reached end of long sequence earlier information was lost . Transformer solved both of these , its a smartly designed neural network , it adds a special layer called ATTENTION and Mlp which allows tokens in a sequence interact with each other. Transformer allows dynamic communication with sequence elements .
---
## Core Architecture :
Transformer consists of a Encoder and decoder both contains two key blocks 1. Attention layer and a 2. Multilayer perceptron (feed forward neural network) . Attention layer is where tokens interact, Mlp is where token refines its representation (privately).The decoder has 3 sub-layers self-attention, cross-attention over encoder output, FFN. That cross-attention is architecturally critical .This helps Transformer for contextual understanding. Also as there is no recurrence and no convolution in our model , in order for the model to make use of the order of the sequence we add positional encodings to the input embeddings at the bottoms of the encoder and decoder stacks.
---

Dataset,Metric,Baseline : They used two data sets WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs and WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary and the metric used was BLEU score ( BiLanguage Evaluation Understudy), the model scored a 28.4 BLEU which was 2 BLEU higher than the previous one in the English-German dataset and single-model state-of-the-art BLEU score of 41.8 on the English-French one . For hardware they used one machine with 8 NVIDIA P100 GPUs. Adam optimizer was used  the learning rate  was varied over the course of training, according to the formula:
 lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
This corresponds to increasing the learning rate linearly for the first training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.
---
## Baseline : 
The previous SOTA(State of the Art) models were ConvS2S and RNN , Transformer beat both ,ConvS2S got 26.36 BLEU on EN-DE, Transformer base got 27.3, big got 28.4. 

---
## My Implementation :
Comparison target: Base model (27.3 BLEU)
*Compute constraints:
1.Hardware: CPU laptop (no GPU)
2.Dataset: Multi30k EN-DE (~29k sentences) instead of WMT 2014 (4.5M sentences)
3.Model size: d_model=256, N=3 layers instead of paper's d_model=512, N=6
4.Training: 10 and 20 epochs instead of 100k steps

## Results:
1. 10 epochs : 
  Final Val loss: 4.14
  BLEU score: 0.42
2. 20 epochs :
   Final Val loss :
   BLEU score :

## Reason for Gap : ~170x less data, half model capacity, fraction of training steps

## Proof of real learning: correctly produces German sentence structure and common vocabulary (e.g. "ein mann mit einem hut" for "a man with a hat")
---

## References:
https://arxiv.org/html/1706.03762v7#S3
https://www.youtube.com/watch?v=avjX3QrYkls
https://www.youtube.com/watch?v=XwYY0lCGWW8

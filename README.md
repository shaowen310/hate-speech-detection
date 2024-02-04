# Twitter Hate Speech Detection

Twitter hate speech detection.

I build data process utilities and PyTorch training skeleton for supervised training. I aim for the codes to be easily transferable to other projects.

## Challenges

The challenges include imbalanced labels, informal language, contextual mentions, label noises, etc. Hate speech detection is also subjective, we may find some confusing labels.

### 1. Imbalanced labels

![Label distribution](https://github.com/shaowen310/hate-speech-detection/blob/main/fig/label_distribution.png?raw=true)

To make the things worse, the number of tweets in the training data is small that it is difficult for deep learning model to learn them from scratch. We may need to try methods optimized for few-shot learning.

### 2. Informal language

There are many spelling errors. Users may use repeated characters to indicate their emotions.

E.g.,

```csv
1259,0,#oitnb bitchezzzz season 4 is here #netflix #yaaay  â¦ 
```

The solutions may let a spelling checker to auto-correct misspelled words. But we should also keep the original spelling in order to identify user emotions.

### 3. Contextual mentions

Users may use abbrevations that are well known for certain circles.

E.g.,

```csv
31966,"is the hp and the cursed child book up for reservations already? if yes, where? if no, when? ððð
```

"hp" here refer to the Harry Potter. We need external knowledge to understand it. So LLM may solve the issue.

### 4. Confusing labels

E.g.,

```csv
1208,1,the latest the maryland daily!  thanks to @user @user #chrismukkah 
```

The user seems not expressing hatred to certain ethnic groups. 

Maybe, we cannot judge the text because we don't know what's in the latest maryland daily. Solutions may require external knowledge base.

## Baseline

I implemented an LSTM baseline. I cleaned up the data (but may also remove some key information), tokenized the texts using an NLTK tokenizer. 

Run `train.py` to train the model.

## Dependency

Run the following code.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy
pip install pandas
```

## Directions to explore

### 1. Experiment large language model for zero-shot or few-shot prompts.

May try prompt engineering first, then instruction-based fine-tuning.

[Respectful or Toxic? Using Zero-Shot Learning with Language Models to Detect Hate Speech](https://github.com/MilaNLProc/prompting_hate_speech)

The reference project's source code is under MIT license.

### 2. Utilizing the info from hash tags. 



This is quite important as many hatred tweets use certain hash tags such as `#white`.

I also want to explore the mentions. However, mentions do not contain much information because they are annonymised to `@user`.

### 3. Emotion and characteristics (needs not annonymised mentions info) detection. 

The way people spell the words may indicate their emotions.

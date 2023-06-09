from data_loader import load_data, tokenizer
from models import BertForMultipleSequenceClassification

from transformers import AutoConfig
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd

label_list = ['확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']

lang = 'spanish'

def train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device):
    
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def eval(model, eval_dataloader, metric, device):
    progress_bar = tqdm(range(len(eval_dataloader)))
    model.eval()
    preds = []
    targets = []
    probs = []
    texts = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
            logits = outputs.logits
            predictions = torch.stack([torch.argmax(logit, dim=-1) for logit in logits], dim=1)
            preds.append(predictions)
            targets.append(batch["labels"])
            text = [tokenizer.convert_ids_to_tokens(x) for x in batch["input_ids"]]
            text = [' '.join(x).replace(' [PAD]', '').replace(' ##', '').replace(' [SEP]', '').replace('[CLS]', '') for x in text]
            texts.extend(text)
            progress_bar.update(1)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    N, M = preds.shape

    res = []
    cn = []
    
    cn.append('text')
    res.append(texts)

    for i in range(M):
        print("%s" % label_list[i], end=' ')
        acc = accuracy_score(targets[:,i], preds[:,i])
        f1 = f1_score(targets[:,i], preds[:,i], average='binary')
        re = recall_score(targets[:,i], preds[:,i], average='binary')
        pre = precision_score(targets[:,i], preds[:,i], average='binary')
        
        cn.append("%s_pred" % label_list[i])
        res.append(preds[:,i])
        cn.append("%s_gt" % label_list[i])
        res.append(targets[:,i])
        
        print('accuracy', acc * 100, end=' ')
        print('f1 score', f1 * 100, end=' ')
        print('precision', pre * 100, end=' ')
        print('recall', re * 100)
    res = pd.DataFrame(zip(*res), columns=cn)
    res.to_csv('%s_result.csv' % lang, index=False)


def main():
    checkpoint = 'bert-base-multilingual-uncased'
    train_dataloader, eval_dataloader = load_data(lang)
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_classes=[2] * 10
    model = BertForMultipleSequenceClassification.from_pretrained(checkpoint, config=config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 6
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device)
    print()

    eval(model, eval_dataloader, 'metric', device)
    model.save_pretrained('./%s_bert' % lang)
    

if __name__ == '__main__':
    main()

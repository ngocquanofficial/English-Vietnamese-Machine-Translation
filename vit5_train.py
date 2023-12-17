from datasets import load_dataset
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
import wandb
from huggingface_hub import login
hf_token = "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"
login(token=hf_token)


# MTET dataset
# dataset = load_dataset("phongmt184172/mtet")
# copy_dataset = load_dataset("phongmt184172/mtet")

# VINAI DATASET
detoken_data_files = {
    "train": "quanpn/PhoMT/PhoMT_json/detokenization/train/train.json",
    "dev": "quanpn/PhoMT/PhoMT_json/detokenization/dev/dev.json",
    "test": "quanpn/PhoMT/PhoMT_json/detokenization/test/test.json"
}

dataset = load_dataset("json", data_files=detoken_data_files, field="data")

# FOR MTET DATASET
# prefix = "translate to Vietnamese: "
# max_length = 128
# def preprocess_function(examples):
#     vi_sentences= []
#     en_sentences = []
#     inputs = []
#     targets = []
#     for prompt in examples["prompt"] :
#         prompt = prompt.split(":")[0]
#         prompt = prompt.lower().split(" ")
#         english = "english"
#         vietnamese = "vietnamese"
#         if english in prompt or "anh" in prompt :
#             en_sentence = "target"
#             vi_sentence = "source"
#         elif vietnamese in prompt or "việt" in prompt :
#             en_sentence = "source"
#             vi_sentence = "target"
#         else :
#             print(prompt)
#         en_sentences.append(en_sentence)
#         vi_sentences.append(vi_sentence)
#     for i, example in enumerate(examples["translation"]) :

#         # JUST FOR CHECKING
#         a = random.randint(1, 100000)
#         if a == 1 :
#             print(example[en_sentences[i]])
#             print(example[vi_sentences[i]])
#             print()


#         input_value = example[en_sentences[i]]
#         target_value = example[vi_sentences[i]]
#         inputs.append(input_value)
#         targets.append(target_value)

#     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#     return model_inputs

# FOR VINAI DATASET
source_lang = "en"
target_lang = "vi"
prefix = "translate to Vietnamese: "
max_length = 128


def preprocess_function(examples):
    inputs = [prefix + example[source_lang]
              for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets,
                             max_length=128, truncation=True)
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


random.seed(23)
dataset_length = len(dataset["train"])
print(dataset_length)

mask = random.sample(range(dataset_length), 500000)
# dataset["train"] = dataset["train"].select(mask)

# Load model directly
checkpoint = "ngocquanofficial/machine_translation_with_V100_second"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=hf_token)

tokenizer_dataset = dataset.map(preprocess_function, batched=True)


model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# All modules in the encoder
modules_to_freeze = [model.encoder.block[i].layer[0]
                     for i in range(len(model.encoder.block))]
# And the decoder modules, which has both a SelfAttention (layer[0])
modules_to_freeze.extend([model.decoder.block[i].layer[0]
                         for i in range(len(model.decoder.block))])
# and CrossAttention (layer[1]) block
modules_to_freeze.extend([model.decoder.block[i].layer[1]
                         for i in range(len(model.decoder.block))])

for module in modules_to_freeze:
    for param in module.parameters():
        param.requires_grad = False  # Actual freezing operation


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
metric = evaluate.load("sacrebleu")


# TRAINING
wandb.login(key="18f117981791693b1b5befd22eb67d03d9bca621")
# model = AutoModelForSeq2SeqLM.from_pretrained("/kaggle/input/checkpoint-14-12-10-epochs-full-dataset/machine_translation_from_huster")
# tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/checkpoint-14-12-10-epochs-full-dataset/machine_translation_from_huster")
# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="machine_translation_PhoMT",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=6,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
    report_to="wandb",
    run_name="ngocquan",
)

# Initialize the W&B integration
wandb.init(project="Machine Translation with PhoMT", config=training_args)

# Create a trainer with the W&B callback
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenizer_dataset["train"],
    eval_dataset=tokenizer_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

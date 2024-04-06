import os
from utils import remove_splchars
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import multiprocessing
import tensorflow as tf
import torch

from simpletransformers.seq2seq import Seq2SeqArgs, Seq2SeqModel

os.environ["KERAS_BACKEND"] = "jax"

def get_data():
    data = ParquetFile("./parquet/train.parquet")
    batch = next(data.iter_batches(batch_size = 1000))
    batch = pa.Table.from_batches([batch]).to_pandas()
    batch.drop(["id"], axis=1, inplace=True)

    #pool = multiprocessing.Pool()
    with multiprocessing.Pool(5) as p:
        batch["article"] = p.map(remove_splchars,batch["article"])


    return batch

def load_ds(data, batch_size):
    values = data.values
    print(values.shape)
    batched = tf.split(values, batch_size, axis=0)


    l = [{"encoder_text": (i[:, 0]), "decoder_text": i[:, 1]} for i in batched]
    return l


data = get_data()
data.columns = ['input_text', 'target_text']


train_ds = load_ds(data, 125)


#Model arguments
model_args = Seq2SeqArgs()
model_args.num_train_epochs = 50
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model, pretrained bart-base, used for finetuning
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-base",
    args=model_args,
    use_cuda=True,
)

#Pretrained bart-large-cnn, uncomment if enough resources are available
# model_large = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name="facebook/bart-large-cnn",
#     args=model_args,
#     use_cuda=True,
# )

model.train_model(data, eval_data=data)

#If computing resources available comment above train and uncomment below
#model_large.train_model(data, eval_data=data)

torch.save(model, './model_finetuned.pth')

#If bart-large-cnn is used uncomment below
#torch.save(model_large, './model_finetuned.pth')
print("Done")

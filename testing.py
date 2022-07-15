from models_t5cosine import T5CosineForNTR
from models_t5vae import T5VAEForNTR
from models_t5 import T5ForNTR
from transformers import T5Tokenizer


# model = T5VAEForNTR.from_pretrained("checkpoints/t5vae.ntr/checkpoint-10000")
# model = T5ForNTR.from_pretrained("checkpoints/t5.ntr/checkpoint-10000")
model = T5CosineForNTR.from_pretrained("checkpoints/t5cosine.ntr/checkpoint-10000")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

context = "Eagles (band) ||| 1994-2001: Reunion, Hell Freezes Over ||| What year did they decide to get back together? ||| 1994. ||| Did all the original members return? ||| The lineup comprised the five Long Run-era members ||| Did they make another album? ||| The ensuing tour spawned a live album ||| was the album successful? ||| debuted at number 1 on the Billboard album chart. ||| Did it have any hit songs? ||| Get Over It"
# ||| Did it win any awards? ||| I don't know." 
question = "Did it win any awards?" 

# NTR
input = tokenizer(context + " ||| " + question, return_tensors='pt')
output = model.generate(**input)
print(tokenizer.decode(output[0]))
# Were there any other hit songs for the Eagles besides Get Over It?"

# NQG
input = tokenizer(context, return_tensors='pt')
output = model.generate(**input)
print(tokenizer.decode(output[0]))

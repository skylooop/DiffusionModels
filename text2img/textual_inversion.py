'''
Textual inversion (based on)
https://colab.research.google.com/drive/1RTHDzE-otzmZOuy8w1WEOxmn9pNcEz3u?usp=sharing#scrollTo=4gVbSkyKdSbE
'''
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_utils import Models
import pyrallis
from dataclasses import dataclass, field
import warnings
import os

warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda")


@dataclass
class params:
    prompt: str = field(default="A picture of a puppy")

def build_mask(input_emb):
    batch_size, seq_len = input_emb.shape[:2]
    causal_attention_mask = Models.text_encoder.text_model._build_causal_attention_mask(batch_size, seq_len, 
                                                                                       input_emb.dtype).to(input_emb.device)
    encoder_outputs = Models.text_encoder.text_model.encoder(
      inputs_embeds=input_emb,
      attention_mask=None,
      causal_attention_mask=causal_attention_mask,
      output_attentions=None,
      output_hidden_states=True, # We want the output embs not the final output
      return_dict=None,
    )
    output = encoder_outputs[0]

    # There is a final layer norm we need to pass these through
    output = Models.text_encoder.text_model.final_layer_norm(output)

    return output
    
def get_pos_embs():
  position_ids = Models.text_encoder.text_model.embeddings.position_ids[:, :77]
  position_embeddings = Models.text_encoder.text_model.embeddings.position_embedding(position_ids)
  return position_embeddings
  
def main(cfg: params):
    
    text_input = Models.tokenizer(cfg.prompt, padding="max_length",
                           max_length=Models.tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
    input_ids = text_input['input_ids'].to(device)
    token_emb = Models.text_encoder.get_input_embeddings()(input_ids)
    replacement_token_embedding = Models.text_encoder.get_input_embeddings()(torch.tensor(2368, device=device))
    token_emb[0, torch.where(input_ids[0]==6829)] = replacement_token_embedding.to(device)    
    
    position_embeddings = get_pos_embs()
    input_embeddings = token_emb + position_embeddings

    #  Feed through to get final output embs
    modified_output_embeddings = build_mask(input_embeddings)

    print(modified_output_embeddings.shape)
    #input_emb = Models.text_encoder.text_model.embeddings(input_ids) # pos + token emb
    
    #final = build_mask(input_emb)


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=params)
    main(cfg)
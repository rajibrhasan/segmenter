from transformers import CLIPTextModel
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", freeze = True):
        super().__init__()
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)

        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, **inputs):
        # text_list: list of strings
        outputs = self.text_encoder(**inputs)
        # Return CLS token representation as text embedding
        return outputs.last_hidden_state
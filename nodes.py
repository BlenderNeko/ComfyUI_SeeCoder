
import folder_paths
import torch
from seecoder.seecoder import SemanticExtractionEncoder, QueryTransformer, Decoder
from seecoder.swin import SwinTransformer
import safetensors.torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.model_management

folder_paths.folder_names_and_paths["seecoder"] = ([os.path.join(folder_paths.models_dir, "seecoders")], folder_paths.supported_ckpt_extensions)

_swine_config = {
    "embed_dim" : 192, 
    "depths" : [ 2, 2, 18, 2 ], 
    "num_heads" : [ 6, 12, 24, 48 ], 
    "window_size" : 12, 
    "ape" : False,
    "drop_path_rate" : 0.3,
    "patch_norm" : True,
}

_decoder_config = {
    "inchannels" : {'res3' : 384, 'res4' : 768, 'res5' : 1536},
    "trans_input_tags" : ['res3', 'res4', 'res5'],
    "trans_dim" : 768,
    "trans_dropout" : 0.1,
    "trans_nheads" : 8,
    "trans_feedforward_dim" : 1024,
    "trans_num_layers" : 6,
}

_qt_config = {
    "in_channels":768,
    "hidden_dim":768,
    "num_queries":[4, 144],
    "nheads":8,
    "num_layers":9,
    "feedforward_dim":2048,
    "pre_norm":False,
    "num_feature_levels":3,
    "enforce_input_project":False,
    "with_fea2d_pos":False
}

class SEECoderImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "seecoder_name": (folder_paths.get_filename_list("seecoder"), ),
            "image": ("IMAGE",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "SEECoderEncode"

    CATEGORY = "conditioning"
        
    def SEECoderEncode(self, seecoder_name, image):
        device = comfy.model_management.get_torch_device()
        path = folder_paths.get_full_path("seecoder", seecoder_name)
        sd = safetensors.torch.load_file(path, device="cpu")
        sd = {k[10:] if k.startswith('ctx.image.') else k: v for k,v in sd.items()}
        is_pa = any([x.startswith("qtransformer.pe_layer") for x in sd.keys()])
        
        swine_config = _swine_config.copy()
        decoder_config = _decoder_config.copy()
        qt_config = _qt_config.copy()
        if is_pa:
            qt_config['with_fea2d_pos'] = True
        
        swine = SwinTransformer(**swine_config)
        decoder = Decoder(**decoder_config)
        queryTransformer = QueryTransformer(**qt_config)

        SEE_encoder = SemanticExtractionEncoder(swine, decoder, queryTransformer)
        SEE_encoder.load_state_dict(sd)
        SEE_encoder = SEE_encoder.to(device)
        SEE_encoder.eval()
        encoding = SEE_encoder(image.movedim(-1,1).to(device)).cpu()

        return ([[encoding, {}]], )
    
class ConcatConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "SEECoderEncode"

    CATEGORY = "_for_testing"
        
    def SEECoderEncode(self, conditioning_to, conditioning_from):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )
    
NODE_CLASS_MAPPINGS = {
    "SEECoderImageEncode": SEECoderImageEncode,
    "ConcatConditioning": ConcatConditioning,
}
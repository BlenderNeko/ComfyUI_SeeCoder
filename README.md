# ComfyUI SeeCoder nodes

This repo contains 2 experimental WIP nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that let's you use [SeeCoders](https://github.com/SHI-Labs/Prompt-Free-Diffusion).

## getting SeeCoders
You can find the seecoders [here](https://huggingface.co/shi-labs/prompt-free-diffusion). They have to be placed at `models/seecoders`

## nodes:

### SEECoderImageEncode

this node can be used to create an embedding from an image

- **image**: the image to encode
- **seecoder_name**: the name of the seecoder

### ConcatConditioning

this node can be used to concat different embeddings together, so you can e.g. create both a text and a visual embedding and concat them together.

- **conditioning_to**: a set of embeddings to concat something to
- **conditioning_from**: the embedding to concat behind those in **conditioning_to**

## TODO:

 - [ ] support for non safetensor formats
 - [ ] bring attention layers in line with ones used in comfy
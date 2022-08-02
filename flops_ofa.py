
def conv(B, kernal_size, out_w, out_h, in_c, out_c):
    return B*kernal_size*kernal_size*out_w*out_h*in_c*out_c

def resnet101(B):
    flops = 0
    flops += conv(B, 7, 240, 240, 3, 64)
    for i in range(3):
        flops += conv(B, 1, 120, 120, 64, 64)
        flops += conv(B, 3, 120, 120, 64, 64)
        flops += conv(B, 1, 120, 120, 64, 256)
    for i in range(4):
        flops += conv(B, 1, 60, 60, 256, 128)
        flops += conv(B, 3, 60, 60, 128, 128)
        flops += conv(B, 1, 60, 60, 128, 512)
    for i in range(23):
        flops += conv(B, 1, 30, 30, 512, 256)
        flops += conv(B, 3, 30, 30, 256, 256)
        flops += conv(B, 1, 30, 30, 256, 1024)
    return flops


def embedding(B, NToken, hidden_dim):
    return B*NToken*hidden_dim*2

def layernorm(B, NToken, hidden_dim):
    return B*NToken*(4*hidden_dim + 1)

def linear(B, NToken, in_dim, out_dim, with_bias=True):
    if with_bias:
        return B*NToken*in_dim*out_dim*2
    else:
        return B*NToken*out_dim*(2*in_dim - 1)

def resnet():
    return 0

def encoder_layer(B, NToken, hidden_size, inter_dim):
    attn_flops = 0
    # WQ, WV, WK
    attn_flops += 3*linear(B, NToken, hidden_size, hidden_size)
    # Attn, Full conneted layer for multihead concat
    attn_flops += 2*linear(B, NToken, NToken, hidden_size) + 2*linear(B, NToken, hidden_size, hidden_size)
    # attn_layernorm
    attn_flops += layernorm(B, NToken, hidden_size)
    # FFN
    attn_flops += linear(B, NToken, inter_dim, hidden_size) + linear(B, NToken, inter_dim, hidden_size)
    # fc1, fc2
    attn_flops += linear(B, NToken, hidden_size, inter_dim) + linear(B, NToken, inter_dim, hidden_size)
    # attn_ln, ffn_layernorm, final_layernorm
    attn_flops += 2*layernorm(B, NToken, hidden_size) + layernorm(B, NToken, inter_dim)
    return attn_flops

def decoder_layer(B, NToken, gtToken, hidden_size, inter_dim):
    attn_flops = 0
    # k_proj, v_proj, q_proj, out_proj
    attn_flops += 4*linear(B, gtToken, hidden_size, hidden_size)
    # self_attn_ln, cross attn ln, self attn layernorm
    attn_flops += layernorm(B, NToken, hidden_size) + 2*layernorm(B, gtToken, hidden_size)
    # encoder_attn
    attn_flops += 2*linear(B, NToken, hidden_size, hidden_size) + 2*linear(B, gtToken, hidden_size, hidden_size)
    # fc1, fc2
    attn_flops += linear(B, gtToken, hidden_size, inter_dim) + linear(B, gtToken, inter_dim, hidden_size)
    # attn_ln, ffn_layernorm, final_layernorm
    attn_flops += 2*layernorm(B, gtToken, hidden_size) + layernorm(B, gtToken, inter_dim)
    return attn_flops

def encoder(B, NToken_txt, hidden_size, inter_dim, NToken_img=900):
    encoder_flops = 0
    # embedding tokens + embed layernorm
    encoder_flops += embedding(B, NToken_txt, hidden_size) + layernorm(B, NToken_txt, hidden_size)
    encoder_flops += resnet()
    # image proj + patch embed layernorm
    encoder_flops += linear(B, NToken_img, 1024, hidden_size) + layernorm(B, NToken_txt, hidden_size)
    layer_num = 6
    for i in range(layer_num):
        encoder_flops += encoder_layer(B, NToken_txt+NToken_img, hidden_size, inter_dim)
    encoder_flops += layernorm(B, NToken_txt+NToken_img, hidden_size)
    return encoder_flops

def decoder(B, NToken, gtToken, hidden_size, inter_dim):
    layer_num = 6
    flops = 0
    for i in range(layer_num):
        flops += decoder_layer(B, NToken, gtToken, hidden_size, inter_dim)
    return flops
print((resnet101(4) + encoder(4, 40, 768, 3072, 900) + decoder(4, 940, 40, 768, 3072))/4e9)
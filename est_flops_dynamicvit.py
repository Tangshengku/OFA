import math
import numpy as np
 
def PatchEmbed(N, P, C):
    return N * P * P * 3 * C

def Head(C):
    return C * 1000

def AuxHead(N, C):
    return N * C * 1000

def Deit_Block(N, C, R=4):
    #return 4 * N * C ** 2 + 2 * N ** 2 * C + 2 * R * N * C ** 2
    return 4 * N * C ** 2 +  N ** 2 * C + (N*0.4) ** 2 * C + 2 * R * N * C ** 2

def Deit_Block_svite(N, C, R=4):
    #C_a=C*1/3
    return 4 * N * C_a * C + 2 * N ** 2 * C_a + 2 * R * N * C ** 2

def PredictorLG(N, C):
    return 5 / 8 * N * C ** 2 + N * C // 2

def DynamicViT(img_size=224, P=16, H=14, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(4):
        #print('i',i)
        #print('before prune N',N)
        blocks += 3 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        #print('after prune N',N)

    head = Head(C)
    return pe + blocks + predictor + head


def Dynamic_Soft_Mask_ViT(img_size=224, P=16, H=14, C=384, sparse=[0.7,0.49,0.343]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(4):
        #print('i',i)
        #print('before prune N',N)
        # if i==0:
        #     blocks += Deit_Block(N, C)
        #     predictor += PredictorLG(N, C)
        #     N = int((N_org - 1) * (1-sparse[i])) + 1  
        #     blocks += 2 * Deit_Block(N, C)

        # else:
            blocks += 3 * Deit_Block(N, C)    #DPDD DDDP DDDP DDD-break
            if i == 3:
                break
            predictor += PredictorLG(N, C)
            #print('sparse[i]',sparse[i])
            N = int((N_org - 1) * (1-sparse[i])) + 1  
        #print('after prune N',N)

    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def Dynamic_Soft_Mask_ViT_3_6_11(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(3):
        if i<2:
            blocks += 3 * Deit_Block(N, C)    #DDDP DDDP DDDD DPD 
            predictor += PredictorLG(N, C)
            #print('sparse[i]',sparse[i])
            N = int((N_org - 1) * (1-sparse[i])) + 1  
        if i==2:
            blocks += 5 * Deit_Block(N, C)
            predictor += PredictorLG(N, C)
            N = int((N_org - 1) * (1-sparse[i])) + 1  
            blocks += Deit_Block(N, C)

        #print('after prune N',N)

    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def Dynamic_Soft_Mask_ViT_3579(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(5):
        print('i',i)
        print('before prune N',N)
        if i==0:
            blocks += 3 * Deit_Block(N, C)
            predictor += PredictorLG(N, C)
            #print('sparse[i]',sparse[i])
            N = int((N_org - 1) * (1-sparse[i])) + 1  

        else:
            #blocks += 2 * Deit_Block(N, C)    #DDDP DDP DDP DDP DDD
            if i == 4:
                blocks += 3 * Deit_Block(N, C) 
                break
            blocks += 2 * Deit_Block(N, C) 
            predictor += PredictorLG(N, C)
            #print('sparse[i]',sparse[i])
            N = int((N_org - 1) * (1-sparse[i])) + 1  
            print('after prune N',N)

    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def Dynamic_Soft_Mask_ViT_4_8(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(3):
        print('i',i)
        print('before prune N',N)
        blocks += 4 * Deit_Block(N, C)   #DDDDP DDDDP DDDD
        if i == 2:
            break
        predictor += PredictorLG(N, C)
        #print('sparse[i]',sparse[i])
        N = int((N_org - 1) * (1-sparse[i])) + 1  
        print('after prune N',N)

    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def Dynamic_Soft_Mask_ViT_6_9(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    # for i in range(3):
    #     print('i',i)
    #     print('before prune N',N)
    #     blocks += 4 * Deit_Block(N, C)  
    #     if i == 2:
    #         break
    #     predictor += PredictorLG(N, C)
    #     #print('sparse[i]',sparse[i])
    #     N = int((N_org - 1) * (1-sparse[i])) + 1  
    #     print('after prune N',N)


    for i in range(4):
        print('i',i)
        print('before prune N',N)
        # if i==0:
        #     blocks += Deit_Block(N, C)
        #     predictor += PredictorLG(N, C)
        #     N = int((N_org - 1) * (1-sparse[i])) + 1  
        #     blocks += 2 * Deit_Block(N, C)

        # else:
        blocks += 3 * Deit_Block(N, C)     #DDD DDDP DDDP DDD
        if i == 3:
            break
        if i>0:
            predictor += PredictorLG(N, C)
            #print('sparse[i]',sparse[i])
        N = int((N_org - 1) * (1-sparse[i])) + 1  
        print('after prune N',N)
    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def Dynamic_Soft_Mask_ViT_1(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0
    for i in range(12):
        print('i',i)
        print('before prune N',N)
        blocks += Deit_Block(N, C)   #DDDDP DDDDP DDDD
        if i == 10:
            break
            predictor += PredictorLG(N, C)
            N = int((N_org - 1) * (1-sparse[i])) + 1  
        print('after prune N',N)

    head = Head(C)
    #print('predictor',predictor/ 1e9)
    #print('all',(pe + blocks + predictor + head)/1e9)
    return pe + blocks + predictor + head

def PatchEmbed4_2(C):
    return 3 * 64 * 49 * 112 ** 2 + 64 ** 2 * 9 * 112 ** 2 * 2 + 64 * C * 112 ** 2

def Dynamic_LV_ViT(img_size=224, P=16, H=14, C=384, rate=1.0, depth=16, mlp_ratio=3):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed4_2(C)
    blocks = 0
    predictor = 0

    for i in range(4):
        blocks += depth // 4 * Deit_Block(N, C, R=mlp_ratio)
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1

    head1 = Head(C)
    aux_head = (N - 1) * C * 1000
    return pe + blocks + predictor + head1 + aux_head


def Dynamic_LV_ViT_Soft(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3,0.3], depth=16, mlp_ratio=3):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed4_2(C)
    blocks = 0
    predictor = 0

    for i in range(4):
        blocks += depth // 4 * Deit_Block(N, C, R=mlp_ratio)
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        N = int((N_org - 1)* (1-sparse[i])) + 1

    head1 = Head(C)
    aux_head = (N - 1) * C * 1000
    #print('predictor',predictor/ 1e9)
    return pe + blocks + predictor + head1 + aux_head


def TokenLearner(img_size=384, P=16, H=12, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H 
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(2):
        #print('i',i)
        print('before prune N',N)
        blocks += 6 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 1:
            break
        #predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        print('after prune N',N)

    head = Head(C)
    return pe + blocks + head #+ predictor



def TokenLearner22(img_size=384, P=16, H=12, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H #+ 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(2):
        #print('i',i)
        print('before prune N',N)
        blocks += 11 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 1:
            break
        #predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        print('after prune N',N)

    head = Head(C)
    return pe + blocks + head #+ predictor

def TokenLearner20(img_size=384, P=16, H=12, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H #+ 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(2):
        #print('i',i)
        print('before prune N',N)
        blocks += 10 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 1:
            break
        #predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        print('after prune N',N)

    head = Head(C)
    return pe + blocks + head #+ predictor

def DeiT12(img_size=224, P=16, H=14, C=384):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 12 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head

def DeiT_Base(img_size=224, P=16, H=14, C=768):  #added
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 12 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head

def LV_ViT(img_size=224, P=16, H=14, C=384, rate=1.0, depth=16, mlp_ratio=3):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed4_2(C)
    blocks = depth * Deit_Block(N, C, R=mlp_ratio)
    head1 = Head(C)
    aux_head = (N - 1) * C * 1000
    return pe + blocks + head1 + aux_head


def AvgPool12(img_size=224, P=16, H=14, C=384):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 6 * Deit_Block(N, C)
    N = (N - 1) // 4 + 1
    blocks += 6 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head


def EViT(img_size=224, P=16, H=14, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(4):
        #print('i',i)
        #print('before prune N',N)
        blocks += 3 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 3:
            break
        #predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        #print('after prune N',N)

    head = Head(C)
    return pe + blocks  + head


def swin_patch_embed():
    B, C, H, W = x.shape
    # FIXME look at relaxing size constraints
    assert H == self.img_size[0] and W == self.img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
    if self.norm is not None:
        x = self.norm(x)

def Swin():

    flops = 0
    flops += swin_patch_embed()
    for i, layer in enumerate(self.layers):
        for blk in self.blocks:

            flops = 0
            H, W = self.input_resolution
            # norm1
            flops += self.dim * H * W
            # W-MSA/SW-MSA
            nW = H * W / self.window_size / self.window_size
            flops += nW * self.attn.flops(self.window_size * self.window_size)
            # mlp
            flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
            # norm2
            flops += self.dim * H * W
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()


        flops += layer.flops()
    flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
    flops += self.num_features * self.num_classes


def print_dynamic_vit():



    #print('EViT Tiny', EViT(C=192, rate=0.7) / 1e9) #change sparse here
    #print('EViT Small', EViT(C=384, rate=0.7) / 1e9) #change sparse here
    #print('EViT Base', EViT(C=768, rate=1) / 1e9) #change sparse here
    #print('EViT Base9', EViT(C=768, rate=0.9) / 1e9) #change sparse here
    #print('EViT Base8', EViT(C=768, rate=0.8) / 1e9) #change sparse here

    
    #print('TokenLearner S/32', TokenLearner(P=32, H=12, C=384, rate=0.05) / 1e9)
    #print('TokenLearner B/16', TokenLearner(P=16, H=24, C=768, rate=0.0138) / 1e9)
    #print('TokenLearner S/32 (22)', TokenLearner22(P=32, H=12, C=384, rate=0.05) / 1e9)
    #print('TokenLearner B/32 (20)', TokenLearner20(P=32, H=12, C=768, rate=0.0138) / 1e9)

    #hello
    #print('DeiT 3/192', DeiT12(C=192) / 1e9)
    #print('DeiT 12/192', DeiT12(C=128) / 1e9)
    #print('DeiT 4/256', DeiT12(C=256) / 1e9)
    #print('DeiT 5/320', DeiT12(C=320) / 1e9)   #78
    print('DeiT 6/384', DeiT12(C=384) / 1e9)   #79.8
    #print('DeiT 7/448', DeiT12(C=448) / 1e9)   #79.8
    #print('DeiT 8/512', DeiT12(C=512) / 1e9)   #79.8

    #print('DeiT 12/768', DeiT_Base(C=768) / 1e9)

    #print('AvgPool 12/384', AvgPool12(C=384) / 1e9)
    #print('####tiny')
    #print('DeiT 12/192', DeiT12(C=192) / 1e9)
    #print('DynamicViT 192/0.8', DynamicViT(C=192, rate=1) / 1e9)

    print('DynamicViT 256/0.7', DynamicViT(C=256, rate=0.7) / 1e9)
    print('DynamicViT 192/0.66', DynamicViT(C=192, rate=0.7) / 1e9)
    print('DynamicViT 192/0.7', DynamicViT(C=192, rate=0.7) / 1e9)
    #print('DynamicViT 256/0.7', DynamicViT(C=256, rate=0.7) / 1e9)
    #print('DynamicViT 320/0.7', DynamicViT(C=320, rate=0.7) / 1e9)
    #print('DynamicViT 384/0.7', DynamicViT(C=384, rate=0.7) / 1e9)
    #print('Dynamic_Soft_Mask_ViT 3_6_11', Dynamic_Soft_Mask_ViT_3_6_11(C=384, sparse=[0.30, 0.6, 0.90]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 192 one predictor', Dynamic_Soft_Mask_ViT_1(C=384, sparse=[0.173]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 192 four predictor', Dynamic_Soft_Mask_ViT_3579(C=192, sparse=[0.091,0.28,0.406,0.657]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 192 two predictor', Dynamic_Soft_Mask_ViT_4_8(C=192, sparse=[0.64,0.80]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 192 two predictor', Dynamic_Soft_Mask_ViT_6_9(C=384, sparse=[0, 0.76,0.88]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 192', Dynamic_Soft_Mask_ViT(C=192, sparse=[0.1530913705583, 0.2089786802030457, 0.4878006091370558]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, sparse=[0,0., 0.95]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 256', Dynamic_Soft_Mask_ViT(C=256, sparse=[0.2429,0.582744,0.7834]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 320', Dynamic_Soft_Mask_ViT(C=320, sparse=[0.5914, 0.8128, 0.914124]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, sparse=[0, 0, 0]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, sparse=[0.1023, 0.1562, 0.3892]) / 1e9) #change sparse here
    #print('####small')
    #print('DynamicViT 384/0.7', DynamicViT(C=384, rate=0.45) / 1e9)
    #print('Dynamic_Soft_Mask_ViT 256', Dynamic_Soft_Mask_ViT(C=256, sparse=[1,1,1]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, sparse=[0,0,0]) / 1e9) #change sparse here
    #print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, 
    #    sparse=[0.4419814141414141,0.6655597989949749,0.7969928]) / 1e9) #change sparse here
    #print('#####base')
    #print('DeiT 12/768', DeiT_Base(C=768) / 1e9)
    #print('DynamicViT 768/0.7', DynamicViT(C=768, rate=0.7) / 1e9)
    #print('DynamicViT 768/0.66', DynamicViT(C=768, rate=0.66) / 1e9)
    #print('Dynamic_Soft_Mask_ViT 768', Dynamic_Soft_Mask_ViT(C=768, sparse=[0.22040161616161616,0.5927597989949749,0.8058344]) / 1e9)

    #print('#####lvvit')
    #print('LV ViT-S', LV_ViT(C=384, depth=16) / 1e9)
    #print('LV ViT-M', LV_ViT(C=512, depth=20) / 1e9)

    #for rate in [0.7, 0.66]:
    #print(f'Dynamic LV ViT-S/0.7', Dynamic_LV_ViT(C=384, depth=16, rate=0.7) / 1e9)
    #print(f'DynamicSoft LV ViT-S', Dynamic_LV_ViT_Soft(C=384, depth=16, 
    #    sparse=[0.3040169696969697,0.6085704522613065,0.7945232]) / 1e9)
    #for rate in [0.8,0.7, 0.66]:
    #print(f'Dynamic LV ViT-M/', Dynamic_LV_ViT(C=512, depth=20, rate=0.59) / 1e9)
    #print(f'DynamicSoft LV ViT-M', Dynamic_LV_ViT_Soft(C=512, depth=20, 
    #    sparse=[0.3758064646464647,0.6985543718592965,0.8199984]) / 1e9)

if __name__ == '__main__':
    print_dynamic_vit()

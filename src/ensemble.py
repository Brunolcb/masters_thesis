import numpy as np
# import SimpleITK as sitk

from .metrics import dice_coef, dice_norm_metric


#função para criar o STAPLE entre as cabeças selecionadas
# def STAPLE_func(pred_bin):
#     total_STAPLES = {number_head:[] for number_head in range(len(pred_bin[0]))}
#     for key in total_STAPLES.keys():
#         for j in range(len(pred_bin)):
#             segs = [pred_bin[j][i].cpu().numpy().reshape(128,128) for i in range(len(pred_bin[0])-1,key-1,-1)]

#             seg_stack = [sitk.GetImageFromArray(seg.astype(np.int16)) for seg in segs]

#             # Run STAPLE algorithm
#             STAPLE_seg_sitk = sitk.STAPLE(seg_stack, 1.0 ) # 1.0 specifies the foreground value

#             # convert back to numpy array
#             STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)
#             total_STAPLES[key].append(STAPLE_seg)
#     return total_STAPLES

#função para carcular a média das saídas da última cabeça até a selecionada
def average_head_func(pred):
    total_mean = {number_head:[] for number_head in range(len(pred[0]))}
    for key in total_mean.keys():
        for j in range(len(pred)):
            choosen_heads = [pred[j][i].cpu().numpy().reshape(128,128) for i in range(len(pred[0])-1,key-1,-1)]
            total_mean[key].append(np.mean(choosen_heads, axis =0))
    return total_mean

#função para calcular as AULAS a partir da ultima até a cabeça selecionada
def aula_head_func(pred):
    aula_heads = {number_head:[] for number_head in range(len(pred[0])-1)}
    for key in aula_heads.keys():
        for j in range(len(pred)):
            aula_curve = [dice_coef(pred[j][i-1].cpu().numpy().reshape(128,128),pred[j][i].cpu().numpy().reshape(128,128)) for i in range(len(pred[0])-1,key,-1)]
            if len(pred[0])-2 > key:
                aula_heads[key].append(np.trapz(aula_curve , dx=1))
            else:
                aula_heads[key].append(aula_curve[0])
        aula_heads[key] = np.array(aula_heads[key])
    return aula_heads

#função para calcular as AULAS com dice norm a partir da ultima até a cabeça selecionada
def aula_ndsc_head_func(pred, r=.079):
    aula_heads = {number_head:[] for number_head in range(len(pred[0])-1)}
    for key in aula_heads.keys():
        for j in range(len(pred)):
            aula_curve = [dice_norm_metric(pred[j][i-1].cpu().numpy().reshape(128,128),pred[j][i].cpu().numpy().reshape(128,128), r=r) for i in range(len(pred[0])-1,key,-1)]
            if len(pred[0])-2 > key:
                aula_heads[key].append(np.trapz(aula_curve , dx=1))
            else:
                aula_heads[key].append(aula_curve[0])
        aula_heads[key] = np.array(aula_heads[key])
    return aula_heads
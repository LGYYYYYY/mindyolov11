import mindspore as ms


def convert_weight(ori_weight, new_weight):
    new_ckpt = []
    param_dict = ms.load_checkpoint(ori_weight)
    for k, v in param_dict.items():
        if '22' in k:
            continue
        new_ckpt.append({'name': k, 'data': v})
    ms.save_checkpoint(new_ckpt, new_weight)


if __name__ == '__main__':
    convert_weight('../pretrain_model/yolov8-l_500e_mAP528-6e96d6bb.ckpt', '../pretrain_model/yolov8l_pretrain.ckpt')
    print('Done!')
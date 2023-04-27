import torch
from torch import nn
import torch.nn.functional as F
tao = nn.Parameter(torch.Tensor([0.04])).cuda()
def downstream_task_forward(model, batch,  criterion, args):
    """
    Example simple function for performing forward pass over a batch input, obtaining predictions and a similarity loss.
    Modify to fit your specific task use case.
    """

    x_txt_cands_mc = batch['text_cands_features']
    y_gt_mc = batch['labels_id']
    video_features = model(batch)

    y_pred  = F.cosine_similarity(video_features.unsqueeze(1), x_txt_cands_mc, dim=-1)  # (N, N_ans)
    loss    = criterion(y_pred / tao, y_gt_mc)
    accs    = (y_pred.argmax(dim=-1) == y_gt_mc).float()

    if args.visible:
        l_y_pred = list(y_pred.argmax(dim=-1).cpu().numpy())
        l_y_gt = list(y_gt_mc.cpu().numpy())
        vids, quess, anss, typee = batch['additional_info']

        for i, (vid, ques, i_pred, i_gt, type) in enumerate(zip(vids, quess, l_y_pred, l_y_gt, typee)):
            try:
                line = f"{vid}\t{ques}\t{anss[i_pred][i]}\t{anss[i_gt][i]}\t{i_gt==i_pred}\t{type}"
            except:
                line = f"{vid}\t{ques}\t{i_pred}\t{i_gt}\t{i_gt == i_pred}\t{type}"
            print(line)
            with open(f'{args.dataset}_{model.config.split}.csv', 'a') as f:
                f.write(line+'\n')

    return loss, accs



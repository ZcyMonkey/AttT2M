import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import random
import sys
import codecs as cs
from tqdm import tqdm
from os.path import join as pjoin
sys.argv = ['GPT_eval_multi.py']
import options.option_transformer as option_trans
args = option_trans.get_args_parser()

args.dataname = 't2m'
#args.resume_pth = './dataset/prepare/pretrained/VQVAE/net_last.pth'
args.resume_pth = './output_spatial/VQVA-spatial/net_last.pth'
args.resume_trans = '/data/zhongchongyang/T2M_GPT/output_GPT_cross_spatial/GPT_cross_spatial/net_best_fid.pth'
#args.resume_trans = '/data/zhongchongyang/T2M_GPT/output_GPT/GPT/net_best_fid.pth'
args.down_t = 2
args.depth = 3
args.block_size = 51
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='/data/zhongchongyang/motiondiffuse')  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


# trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
#                                               embed_dim=1024,
#                                               clip_dim=args.clip_dim,
#                                               block_size=args.block_size,
#                                               num_layers=9,
#                                               n_head=16,
#                                               drop_out_rate=args.drop_out_rate,
#                                               fc_rate=args.ff_rate)
trans_encoder = trans.Text2Motion_Cross_Transformer(num_vq=args.nb_code,
                                              embed_dim=1024,
                                              clip_dim=args.clip_dim,
                                              block_size=args.block_size,
                                              num_layers=9,
                                              n_head=16,
                                              drop_out_rate=args.drop_out_rate,
                                              fc_rate=args.ff_rate,
                                              num_layers_cross=args.num_layers_cross)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

print ('loading transformer checkpoint from {}'.format(args.resume_trans))
ckpt = torch.load(args.resume_trans, map_location='cpu')
trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()

mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()
input_text = "a person walks up to a backwards chair and sits down on it with legs outstretched, then stands back up"
output_type = "smpl"
outdir = './smpl_viz_new/'
split_file = pjoin('/data/zhongchongyang/TTM/HumanML3D-main/HumanML3D', 'test.txt')
motion_dir = pjoin('/data/zhongchongyang/TTM/HumanML3D-main/HumanML3D', 'new_joint_vecs')
text_dir = pjoin('/data/zhongchongyang/TTM/HumanML3D-main/HumanML3D', 'texts')
id_list = []
text_data = []
# with cs.open(split_file, 'r') as f:
#     for line in f.readlines():
#         id_list.append(line.strip())
#
# for name in tqdm(id_list):
#     try:
#         motion = np.load(pjoin(motion_dir, name + '.npy'))
#         if (len(motion)) < 40 or (len(motion) >= 200):
#             continue
#
#         flag = False
#         with cs.open(pjoin(text_dir, name + '.txt')) as f:
#             for line in f.readlines():
#                 text_dict = {}
#                 line_split = line.strip().split('#')
#                 caption = line_split[0]
#                 tokens = line_split[1].split(' ')
#                 f_tag = float(line_split[2])
#                 to_tag = float(line_split[3])
#                 f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                 to_tag = 0.0 if np.isnan(to_tag) else to_tag
#
#                 text_dict = caption
#                 if f_tag == 0.0 and to_tag == 0.0:
#                     flag = True
#                     text_data.append(text_dict)
#     except Exception as e:
#         # print(e)
#         pass




# clip_text = ["a man walks up one set of steps to the right, then turn right and walks uo another set of steps and then turns right again and walks across platform"]
# clip_text.append("a person walks in the shape of a j but runs the last two step and heads back to where they started")
# clip_text = ["the person is in a sitting position with his arms in front them when they raise their arms out to the side, lower them, and raise their arms again before bringing them back to their original position."]
# clip_text.append("a person raises and lowers their left hand.")
# clip_text.append("a person stands while moving their left arm as if eating something with a spoon or fork three times.")
# clip_text.append("the figure takes a seat, appears to make a throwing motion, and then stands up.")
# clip_text.append("a person taking something from a shelf with their right hand")
# clip_text.append("the person is picking something up and putting it on something.")
# clip_text.append("person runs forward and then jumps forwards with both feet and lands on two feet")
# clip_text.append("a person brings their left arm towards their face")
# clip_text.append("a person holds their hands at their waist while making pecking motions with their body like a chicken.")
# clip_text.append("person is bent over wiping a table")
# clip_text.append("the man puts the box down and runs")
# clip_text.append("a man reaches forward, pulls back a lever with his left hand, and then he shuts a door with his right hand.")
# clip_text.append("a person places something on the ground.")
# clip_text.append("a person picks something up with their right hand, walks to the left, and then drys something off.")
# clip_text.append("a person walks forward, hops backwards, then defends themselves by putting their hands up in defense")
# clip_text.append("a person walks turning to the right.")
# clip_text.append("a person standing in place lifts and waves with his right hand.")
# clip_text.append("a person stomps with their left foot.")
# clip_text.append("the person uses the left to grab the right elbow and swing it. the right arm raises up.")
# clip_text.append("a person walking straight forward.")
# clip_text.append("someone rubs their belly with their left hand and rubs their head with their right hand at the same time.")
# clip_text.append("person opens a drink, takes a sip, and walks backward")
# clip_text.append("a person steps backwards, then sits down, gets back up and walks to the left.")
# clip_text.append("a person has their left hand by their face and is gesturing that their head is dizzy.")
# clip_text.append("the person steps a little wider than shoulder width apart first with their left foot, then with their right before squatting 4 times.")
# clip_text.append("a person standing in the same spot while rocking sideways.")
# clip_text.append("a person holds right hand over left forearm, releases hand, and stands up straight with arms by his side.")
# clip_text.append("a person standing straight ,holding hands .")
# clip_text.append("a person walks turning to the left.")
# clip_text.append("a person walks in a counterclockwise circle while keeping their body facing forward.")

clip_text = ["a person is jumping to the right"]
clip_text.append("a person is kicking his left leg")
clip_text.append("a person is kicking his right leg")
clip_text.append("a person is waving his left arm")
clip_text.append("a person is waving his right arm")
clip_text.append("a person is kicking before puching")
clip_text.append("a person is kicking after puching")
clip_text.append("a person is waking and turning right")
clip_text.append("a person is waking and turning left")
clip_text.append("a person is jumping to the left")
for j in range(10):
    #clip_text = random.choice(text_data)
    #clip_text = "someone steps back with their right foot and then sits down while placing his hands on his knees with elbows out"
    text = clip.tokenize(clip_text[j], truncate=True).cuda()
    word_emb = clip_model.token_embedding(text).type(clip_model.dtype)
    word_emb = word_emb + clip_model.positional_embedding.type(clip_model.dtype)
    word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
    word_emb = clip_model.transformer(word_emb)
    word_emb = clip_model.ln_final(word_emb).permute(1, 0, 2).float()
    feat_clip_text = clip_model.encode_text(text).float()
    from utils.motion_process import recover_from_ric
    import visualization.plot_3d_global as plot_3d
    from render_final import render
    for i in range(6):
        index_motion = trans_encoder.sample(feat_clip_text[0:1],word_emb[0:1], True)
        #index_motion = trans_encoder.sample(feat_clip_text[0:1], True)
        pred_pose = net.forward_decoder(index_motion)


        pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)
        pred_xyz = pred_xyz.squeeze(0).detach().cpu().numpy()
        if output_type =="smpl":
            print('pred', pred_xyz.shape, 'visualizing:'+clip_text[j])
            if not os.path.exists(outdir + clip_text[j]):
                os.makedirs(outdir + clip_text[j])
            render(pred_xyz, outdir=outdir + clip_text[j]+'/', device_id=0, name=str(i), pred=True)
        elif output_type =="skeleton":
            pose_vis = plot_3d.draw_to_batch(xyz.detach().cpu().numpy(),clip_text, ['clockwise_'+str(i)+'.gif'])
        else:
            print("Wrong output type")
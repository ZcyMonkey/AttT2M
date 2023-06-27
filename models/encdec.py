import torch
import math
import torch.nn as nn
from models.resnet import Resnet1D
t2m_kinematic_chain_for_train = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21,22], [9, 13, 16, 18, 20,22]]
kit_kinematic_chain_for_train = [[0, 11, 12, 13, 14, 15,21], [0, 16, 17, 18, 19, 20,21], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]]
class PositionalEncoding(nn.Module):
    def __init__(self, src_dim, embed_dim, dropout, max_len=100, hid_dim=512):
        """
        :param src_dim:  orignal input dimension
        :param embed_dim: embedding dimension
        :param dropout: dropout rate
        :param max_len: max length
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(src_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, embed_dim)
        self.relu = nn.ReLU()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embed_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / embed_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, step=None):
        """
        :param input: L x N x D
        :param step:
        :return:
        """
        # raw_shape = input.shape[:-2]
        # j_num, f_dim = input.shape[-2], input.shape[-1]
        # input = input.reshape(-1, j_num, f_dim).transpose(0, 1)
        emb = self.linear2(self.relu(self.linear1(input)))
        emb = emb * math.sqrt(self.embed_dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        # emb = emb.transpose(0, 1).reshape(raw_shape + (j_num, -1))
        return emb
class Abstract_Transformer(nn.Module):

    def __init__(self, transformer_layers,
                 transformer_latents,
                 transformer_ffsize,
                 transformer_heads,
                 transformer_dropout,
                 transformer_srcdim,
                 correspondence,
                 njoints,
                 activation="gelu"):
        super(Abstract_Transformer, self).__init__()


        self.correspondence = correspondence
        self.nparts = len(correspondence)
        self.njoints = njoints # append root volocity

        self.num_layers = transformer_layers
        self.latent_dim = transformer_latents
        self.ff_size = transformer_ffsize
        self.num_heads = transformer_heads

        self.dropout = transformer_dropout
        self.src_dim = transformer_srcdim
        self.activation = activation



        self.joint_pos_encoder = PositionalEncoding(self.src_dim, self.latent_dim, self.dropout)

        spaceTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)

        self.spatialTransEncoder = nn.TransformerEncoder(spaceTransEncoderLayer,
                                                         num_layers=self.num_layers)

        self.parameter_part = nn.Parameter(torch.randn(self.nparts, 1, self.latent_dim) * 0.1)



    def forward(self, x, attention_mask,offset=None):


        b, t, j,c= x.shape[0], x.shape[1], x.shape[2],x.shape[3]

        x = x.transpose(0, 2).reshape(j,t*b,c)  # J BT E

        encoding = self.joint_pos_encoder(x)

        encoding_app = torch.cat((self.parameter_part.repeat(1, encoding.shape[1], 1), encoding), dim=0)

        # key_padding_mask = get_key_padding_mask(self.correspondence, self.njoints, b_size)
        final = self.spatialTransEncoder(encoding_app, mask=attention_mask)

        final_parts = final[:self.nparts].reshape(self.nparts,t,b,-1).transpose(0,2) # B * nparts E * T

        #final_parts = final_parts.reshape(b_size, -1, t_size)

        return final_parts


class Encoder_spatial(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.spatial_transformer = Abstract_Transformer(transformer_layers=2,
                                                        transformer_latents=128,
                                                        transformer_ffsize=512,
                                                        transformer_heads=4,
                                                        transformer_dropout= 0.2,
                                                        transformer_srcdim=32,
                                                        correspondence=t2m_kinematic_chain_for_train,
                                                        njoints=23
                                                        )
        self.root_joint_embed = nn.Linear(7, 32)
        self.other_joint_embed = nn.Linear(12, 32)
        self.contact_embed = nn.Linear(4, 32)
    def get_transformer_matrix(self,part_list, njoints):
        """
        :param part_list: body part list  [[0 ,1 , 2], [1]]   n * 4
        :param njoints: body joints' number plus root velocity
        :return:
        """
        nparts = len(part_list)
        matrix = torch.zeros([nparts + njoints, njoints])

        for i in range(nparts):
            matrix[i, part_list[i]] = 1
            for j in part_list[i]:
                for k in part_list[i]:
                    matrix[j + nparts, k] = 1
        matrix[:, 0] = 1
        matrix[:, -1] = 1

        matrix = torch.cat((torch.zeros([njoints + nparts, nparts]), matrix), dim=1)
        for p in range(nparts + njoints):
            matrix[p, p] = 1

        matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
        return matrix

    def forward(self, x):
        x = x.permute(0,2,1).float()
        B, T = x.shape[0], x.shape[1]
        root_feature = torch.cat((x[:,:,:4],x[:,:,193:196]),dim=2).unsqueeze(2)
        other_joint = torch.cat((x[:,:,4:193],x[:,:,196:259]),dim=2)
        position = other_joint[:,:,:63].reshape(B, T,21,3)
        rotation = other_joint[:,:,63:189].reshape(B, T,21,6)
        velocity = other_joint[:,:,189:].reshape(B, T,21,3)
        other_joint_feature = torch.cat((torch.cat((position,rotation),dim=3),velocity),dim=3)
        contact = x[:,:,259:].unsqueeze(2)
        root_feature = self.root_joint_embed(root_feature)
        other_joint_feature = self.other_joint_embed(other_joint_feature)
        contact_feature = self.contact_embed(contact)

        h = torch.cat((torch.cat((root_feature,other_joint_feature),dim=2),contact_feature),dim=2)
        attention_mask = self.get_transformer_matrix(t2m_kinematic_chain_for_train, 23).to(x.device)
        h = self.spatial_transformer(h,attention_mask).reshape(B,T,5*128)
        h = h.permute(0,2,1)
        out = self.model(h)
        return out

class Encoder_spatial_kit(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.spatial_transformer = Abstract_Transformer(transformer_layers=2,
                                                        transformer_latents=128,
                                                        transformer_ffsize=512,
                                                        transformer_heads=4,
                                                        transformer_dropout= 0.2,
                                                        transformer_srcdim=32,
                                                        correspondence=t2m_kinematic_chain_for_train,
                                                        njoints=22
                                                        )
        self.root_joint_embed = nn.Linear(7, 32)
        self.other_joint_embed = nn.Linear(12, 32)
        self.contact_embed = nn.Linear(4, 32)
    def get_transformer_matrix(self,part_list, njoints):
        """
        :param part_list: body part list  [[0 ,1 , 2], [1]]   n * 4
        :param njoints: body joints' number plus root velocity
        :return:
        """
        nparts = len(part_list)
        matrix = torch.zeros([nparts + njoints, njoints])

        for i in range(nparts):
            matrix[i, part_list[i]] = 1
            for j in part_list[i]:
                for k in part_list[i]:
                    matrix[j + nparts, k] = 1
        matrix[:, 0] = 1
        matrix[:, -1] = 1

        matrix = torch.cat((torch.zeros([njoints + nparts, nparts]), matrix), dim=1)
        for p in range(nparts + njoints):
            matrix[p, p] = 1

        matrix = matrix.float().masked_fill(matrix == 0., float(-1e20)).masked_fill(matrix == 1., float(0.0))
        return matrix

    def forward(self, x):
        x = x.permute(0,2,1).float()
        B, T = x.shape[0], x.shape[1]
        root_feature = torch.cat((x[:,:,:4],x[:,:,184:187]),dim=2).unsqueeze(2)
        other_joint = torch.cat((x[:,:,4:184],x[:,:,187:247]),dim=2)
        position = other_joint[:,:,:60].reshape(B, T,20,3)
        rotation = other_joint[:,:,60:180].reshape(B, T,20,6)
        velocity = other_joint[:,:,180:].reshape(B, T,20,3)
        other_joint_feature = torch.cat((torch.cat((position,rotation),dim=3),velocity),dim=3)
        contact = x[:,:,247:].unsqueeze(2)
        root_feature = self.root_joint_embed(root_feature)
        other_joint_feature = self.other_joint_embed(other_joint_feature)
        contact_feature = self.contact_embed(contact)

        h = torch.cat((torch.cat((root_feature,other_joint_feature),dim=2),contact_feature),dim=2)
        attention_mask = self.get_transformer_matrix(kit_kinematic_chain_for_train, 22).to(x.device)
        h = self.spatial_transformer(h,attention_mask).reshape(B,T,5*128)
        h = h.permute(0,2,1)
        out = self.model(h)
        return out










class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)




class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    

from models.blocks import *
import torch.nn.functional as F
import numpy as np
from models.gcn import GCN
from lib.utils import square_distance


class KPFCNN(nn.Module):

    def __init__(self, config):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############
        # Current radius of convolution and feature dimension
        # 卷积的当前半径和特征维度
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        # 1(modelnet40)
        in_dim = config.in_feats_dim
        # 512(modelnet40)
        out_dim = config.first_feats_dim
        # 15(modelnet40)
        self.K = config.num_kernel_points
        self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
        # 96(modelnet40)
        self.final_feats_dim = config.final_feats_dim
        # True(modelnet40)
        self.condition = config.condition_feature
        # True(modelnet40)
        self.add_cross_overlap = config.add_cross_score

        #####################
        # List Encoder blocks
        #####################
        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks 连续块上循环
        for block_i, block in enumerate(config.architecture):

            # Check equivariance 检查是否符合标准
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection 检测更改为跳跃连接的下一层
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck layer and GNN part
        #####################
        # 256(modelnet40)
        gnn_feats_dim = config.gnn_feats_dim
        self.bottle = nn.Conv1d(in_dim, gnn_feats_dim,kernel_size=1,bias=True)
        # 10(modelnet40)
        k=config.dgcnn_k
        # 4(modelnet40)
        num_head = config.num_head
        # nets:['self','cross','self']
        self.gnn = GCN(num_head,gnn_feats_dim, k, config.nets)
        self.proj_gnn = nn.Conv1d(gnn_feats_dim, gnn_feats_dim, kernel_size=1, bias=True)
        self.proj_score = nn.Conv1d(gnn_feats_dim, 1, kernel_size=1, bias=True)

        
        #####################
        # List Decoder blocks
        #####################
        if self.add_cross_overlap:
            out_dim = gnn_feats_dim + 2
        else:
            out_dim = gnn_feats_dim + 1

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block 找到第一个下采样块
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break
        
        # Loop over consecutive blocks 在连续的块上循环
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat 跳跃联接concat增加维度
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2
        return

    def regular_score(self,score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

# batch是个什么东西
    def forward(self, batch):
        # Get input features 获得输入特征
        x = batch['features'].clone().detach()
        len_src_c = batch['stack_lengths'][-1][0]
        len_src_f = batch['stack_lengths'][0][0]
        pcd_c = batch['points'][-1]
        pcd_f = batch['points'][0]
        src_pcd_c, tgt_pcd_c = pcd_c[:len_src_c], pcd_c[len_src_c:]

        sigmoid = nn.Sigmoid()
        #################################
        # 1. joint encoder part 联合编码器部分
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)
        
        #################################
        # 2. project the bottleneck features
        feats_c = x.transpose(0,1).unsqueeze(0)  #[1, C, N]
        feats_c = self.bottle(feats_c)  #[1, C, N]
        unconditioned_feats = feats_c.transpose(1,2).squeeze(0)

        #################################
        # 3. apply GNN to communicate the features and get overlap score 应用GNN传递特征并且获得重叠分数
        src_feats_c, tgt_feats_c = feats_c[:,:,:len_src_c], feats_c[:,:,len_src_c:]
        src_feats_c, tgt_feats_c= self.gnn(src_pcd_c.unsqueeze(0).transpose(1,2), tgt_pcd_c.unsqueeze(0).transpose(1,2),src_feats_c, tgt_feats_c)
        feats_c = torch.cat([src_feats_c, tgt_feats_c], dim=-1)

        feats_c = self.proj_gnn(feats_c)   
        # 重叠分数
        scores_c =  self.proj_score(feats_c)

        feats_gnn_norm = F.normalize(feats_c, p=2, dim=1).squeeze(0).transpose(0,1)  #[N, C]
        feats_gnn_raw = feats_c.squeeze(0).transpose(0,1)
        scores_c_raw = scores_c.squeeze(0).transpose(0,1) #[N, 1]
       
        ####################################
        # 4. decoder part
        src_feats_gnn, tgt_feats_gnn = feats_gnn_norm[:len_src_c], feats_gnn_norm[len_src_c:]
        inner_products = torch.matmul(src_feats_gnn, tgt_feats_gnn.transpose(0,1))

        src_scores_c, tgt_scores_c = scores_c_raw[:len_src_c], scores_c_raw[len_src_c:]
        # temperature parameter
        temperature = torch.exp(self.epsilon) + 0.03
        s1 = torch.matmul(F.softmax(inner_products / temperature ,dim=1) ,tgt_scores_c)
        s2 = torch.matmul(F.softmax(inner_products.transpose(0,1) / temperature,dim=1),src_scores_c)
        # 显著性分数
        scores_saliency = torch.cat((s1,s2),dim=0)
        
        if(self.condition and self.add_cross_overlap): 
            x = torch.cat([scores_c_raw,scores_saliency,feats_gnn_raw], dim=1)
        elif(self.condition and not self.add_cross_overlap):
            x = torch.cat([scores_c_raw,feats_gnn_raw], dim=1)
        elif(not self.condition and self.add_cross_overlap):
            x = torch.cat([scores_c_raw, scores_saliency, unconditioned_feats], dim = 1)
        elif(not self.condition and not self.add_cross_overlap):
            x = torch.cat([scores_c_raw, unconditioned_feats], dim = 1)
    
        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)
        feats_f = x[:,:self.final_feats_dim]
        scores_overlap = x[:, self.final_feats_dim]
        scores_saliency = x[:, self.final_feats_dim+1]

        # safe guard our score
        scores_overlap = torch.clamp(sigmoid(scores_overlap.view(-1)),min=0,max=1)
        scores_saliency = torch.clamp(sigmoid(scores_saliency.view(-1)),min=0,max=1)
        scores_overlap = self.regular_score(scores_overlap)
        scores_saliency = self.regular_score(scores_saliency)

        # normalise point-wise features
        feats_f = F.normalize(feats_f, p=2, dim=1)

        return feats_f, scores_overlap, scores_saliency

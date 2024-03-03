import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import mindspore as ms

class A_2_net_Loss(nn.Module):
    def __init__(self, code_length, gamma, batch_size, margin=1, finetune=False):
        super(A_2_net_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma
        self.MSE_loss = nn.MSELoss()
        self.batch_size = batch_size
        self.margin = margin
        self.finetune = finetune

    #First train by following ADSH(use hash_loss & quantization_loss) and fine-tune with reconstruction_loss and decorrelation_loss
    def forward(self, F, B, S, omega, dret, all_f, deep_S, inputs):
        #print(F.shape, B.shape, S.shape,dret.shape,all_f.shape,deep_S.shape)

        I = torch.eye(self.code_length, device=F.device)
        decorrelation_loss = torch.nn.functional.mse_loss(F.t() @ F, self.batch_size * I, reduction='mean')/self.batch_size/self.code_length*self.margin
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12*self.margin
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12*self.margin

        if self.finetune:
            gan_loss = torch.nn.functional.mse_loss(inputs['_img'], inputs['img'])*0.1
            gan_loss2 = torch.nn.functional.mse_loss(inputs['_img2'], inputs['img'])*0.1
            #gan_loss += gan_loss2
        else:
            gan_loss = torch.nn.functional.mse_loss(inputs['_img'], inputs['img'])*0.01
        #gan_loss = I.new_tensor(0.)

        if self.finetune:
            reconstruction_loss = self.MSE_loss(dret, all_f)
            loss = hash_loss + quantization_loss + reconstruction_loss + decorrelation_loss
        else:
            with torch.no_grad():
                reconstruction_loss = self.MSE_loss(dret, all_f)
            loss = hash_loss + quantization_loss + decorrelation_loss

        return loss, hash_loss, quantization_loss, reconstruction_loss, decorrelation_loss, gan_loss

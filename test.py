from guided_diffusion.model import UNetModel_V
from guided_diffusion.unet import UNetModel_newpreview
import torchsummary

if __name__ == "__main__":
    # model = UNetModel_V(image_size=256,
    #                     in_channels=4,
    #                     model_channels=128,
    #                     out_channels=1,
    #                     num_res_blocks=2,
    #                     attention_resolutions=(16, 8),
    #                     num_heads=1,
    #                     )
    model = UNetModel_newpreview(image_size=256,
                                 in_channels=4,
                                 model_channels=128,
                                 out_channels=1,
                                 num_res_blocks=2,
                                 attention_resolutions=(16, 8),
                                 num_heads=1,
                                 )
    
    # print(torchsummary.summary(model, input_size=(4, 256, 256)))
    print(model)
    print(sum(p.numel() for p in model.parameters()))
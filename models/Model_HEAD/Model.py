import multiprocessing
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

class HEADModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()

        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb

        suggest_batch_size = 4

        yn_str = {True:'y',False:'n'}
        default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', False)
        default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', 64)
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', 16)
        default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            self.ask_autobackup_hour()
            self.ask_write_preview_history()
            self.ask_target_iter()
            self.ask_batch_size(suggest_batch_size)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2

            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if len(device_config.devices) == 1:
                self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 10.0", help_message="Train the network in Generative Adversarial manner. Accelerates the speed of training. Forces the neural network to learn small details of the face. You can enable/disable this option at any time. Typical value is 1.0"), 0.0, 10.0 )

            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        self.model_data_format = "NCHW" if len(device_config.devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        conv_kernel_initializer = nn.initializers.ca()

        class Downscale(nn.ModelBase):
            def __init__(self, in_ch, out_ch, kernel_size=5, dilations=1, subpixel=True, use_activator=True, *kwargs ):
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.kernel_size = kernel_size
                self.dilations = dilations
                self.subpixel = subpixel
                self.use_activator = use_activator
                super().__init__(*kwargs)

            def on_build(self, *args, **kwargs ):
                self.conv1 = nn.Conv2D( self.in_ch,
                                        self.out_ch // (4 if self.subpixel else 1),
                                        kernel_size=self.kernel_size,
                                        strides=1 if self.subpixel else 2,
                                        padding='SAME', dilations=self.dilations, kernel_initializer=conv_kernel_initializer)

            def forward(self, x):
                x = self.conv1(x)
                if self.subpixel:
                    x = nn.space_to_depth(x, 2)
                if self.use_activator:
                    x = tf.nn.leaky_relu(x, 0.1)
                return x

            def get_out_ch(self):
                return (self.out_ch // 4) * 4

        class DownscaleBlock(nn.ModelBase):
            def on_build(self, in_ch, ch, n_downscales, kernel_size, dilations=1, subpixel=True):
                self.downs = []

                last_ch = in_ch
                for i in range(n_downscales):
                    cur_ch = ch*( min(2**i, 8)  )
                    self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size, dilations=dilations, subpixel=subpixel) )
                    last_ch = self.downs[-1].get_out_ch()

            def forward(self, inp):
                x = inp
                for down in self.downs:
                    x = down(x)
                return x

        class Upscale(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, x):
                x = self.conv1(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = nn.depth_to_space(x, 2)
                return x

        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = tf.nn.leaky_relu(inp + x, 0.2)
                return x

        class UpdownResidualBlock(nn.ModelBase):
            def on_build(self, ch, inner_ch, kernel_size=3 ):
                self.up   = Upscale (ch, inner_ch, kernel_size=kernel_size)
                self.res  = ResidualBlock (inner_ch, kernel_size=kernel_size)
                self.down = Downscale (inner_ch, ch, kernel_size=kernel_size, use_activator=False)

            def forward(self, inp):
                x = self.up(inp)
                x = upx = self.res(x)
                x = self.down(x)
                x = x + inp
                x = tf.nn.leaky_relu(x, 0.2)
                return x, upx

        class Encoder(nn.ModelBase):
            def on_build(self, in_ch, e_ch):
                self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=5, kernel_size=5, dilations=1, subpixel=False)

            def forward(self, inp):
                x = nn.flatten(self.down1(inp))
                return x

        class Inter(nn.ModelBase):
            def __init__(self, in_ch, lowest_dense_res, ae_ch, ae_out_ch, **kwargs):
                self.in_ch, self.lowest_dense_res, self.ae_ch, self.ae_out_ch = in_ch, lowest_dense_res, ae_ch, ae_out_ch
                super().__init__(**kwargs)

            def on_build(self):
                in_ch, lowest_dense_res, ae_ch, ae_out_ch = self.in_ch, self.lowest_dense_res, self.ae_ch, self.ae_out_ch

                self.dense1 = nn.Dense( in_ch, ae_ch )
                self.dense2 = nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                self.upscale1 = Upscale(ae_out_ch, ae_out_ch*2)
                
            def forward(self, inp):
                x = self.dense1(inp)
                x = self.dense2(x)
                x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                x = self.upscale1(x)
                return x

            def get_out_ch(self):
                return self.ae_out_ch

        class Decoder(nn.ModelBase):
            def on_build(self, in_ch, d_ch, d_mask_ch ):

                self.upscale0 = Upscale(in_ch, d_ch*8, kernel_size=3)
                self.upscale1 = Upscale(d_ch*8, d_ch*4, kernel_size=3)
                self.upscale2 = Upscale(d_ch*4, d_ch*2, kernel_size=3)
                self.upscale3 = Upscale(d_ch*2, d_ch*1, kernel_size=3)
                
                self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                self.res2 = ResidualBlock(d_ch*2, kernel_size=3)
                self.res3 = ResidualBlock(d_ch*1, kernel_size=3)
                self.out_conv  = nn.Conv2D( d_ch*1, 3, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)

                self.upscalem0 = Upscale(in_ch, d_mask_ch*8, kernel_size=3)
                self.upscalem1 = Upscale(d_mask_ch*8, d_mask_ch*4, kernel_size=3)
                self.upscalem2 = Upscale(d_mask_ch*4, d_mask_ch*2, kernel_size=3)
                self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=3)
                self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)
                
            """
            def get_weights_ex(self, include_mask):
                # Call internal get_weights in order to initialize inner logic
                self.get_weights()

                weights = self.upscale0.get_weights() + self.upscale1.get_weights() + self.upscale2.get_weights() \
                          + self.res0.get_weights() + self.res1.get_weights() + self.res2.get_weights() + self.out_conv.get_weights()

                if include_mask:
                    weights += self.upscalem0.get_weights() + self.upscalem1.get_weights() + self.upscalem2.get_weights() \
                               + self.out_convm.get_weights()
                return weights
                
            """
            def get_weights_ex(self, include_mask):
                # Call internal get_weights in order to initialize inner logic
                self.get_weights()

                weights = self.upscale0.get_weights() + self.upscale1.get_weights() + self.upscale2.get_weights() + self.upscale3.get_weights()\
                          + self.res0.get_weights() + self.res1.get_weights() + self.res2.get_weights() + self.res3.get_weights() + self.out_conv.get_weights()

                if include_mask:
                    weights += self.upscalem0.get_weights() + self.upscalem1.get_weights() + self.upscalem2.get_weights() + self.upscalem3.get_weights() \
                               + self.out_convm.get_weights()
                return weights
                

            def forward(self, inp):
                z = inp

                x = self.upscale0(z)
                x = self.res0(x)
                x = self.upscale1(x)
                x = self.res1(x)
                x = self.upscale2(x)
                x = self.res2(x)
                x = self.upscale3(x)
                x = self.res3(x)
                
                m = self.upscalem0(z)
                m = self.upscalem1(m)
                m = self.upscalem2(m)
                m = self.upscalem3(m)
                return tf.nn.sigmoid(self.out_conv(x)), \
                       tf.nn.sigmoid(self.out_convm(m))
                       
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.resolution = resolution = 448
        self.learn_mask = learn_mask = True
        eyes_prio = True
        ae_dims = self.options['ae_dims']
        e_dims = self.options['e_dims']
        d_dims = self.options['d_dims']
        d_mask_dims = self.options['d_mask_dims']
        self.pretrain = False
        self.pretrain_just_disabled = False
        if self.pretrain_just_disabled:
            self.set_iter(0)

        self.gan_power = gan_power = self.options['gan_power'] if not self.pretrain else 0.0

        masked_training = True

        models_opt_on_gpu = False if len(devices) == 0 else True if len(devices) > 1 else self.options['models_opt_on_gpu']
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch = 3
        output_ch = 3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        lowest_dense_res = resolution // 32

        self.model_filename_list = []


        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_srcm_all = tf.placeholder (nn.floatx, mask_shape)
            self.target_dstm_all = tf.placeholder (nn.floatx, mask_shape)
            
        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.compute_output_channels ( (nn.floatx, bgr_shape))

            self.inter = Inter (in_ch=encoder_out_ch, lowest_dense_res=lowest_dense_res, ae_ch=ae_dims, ae_out_ch=ae_dims, name='inter')
            inter_out_ch = self.inter.compute_output_channels ( (nn.floatx, (None,encoder_out_ch)))

            self.decoder_src = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_src')
            self.decoder_dst = Decoder(in_ch=inter_out_ch, d_ch=d_dims, d_mask_ch=d_mask_dims, name='decoder_dst')

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                            [self.inter,       'inter.npy'      ],
                                            [self.decoder_src, 'decoder_src.npy'],
                                            [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                if gan_power != 0:
                    self.D_src = nn.PatchDiscriminator(patch_size=resolution//16, in_ch=output_ch, base_ch=256, name="D_src")
                    self.D_dst = nn.PatchDiscriminator(patch_size=resolution//16, in_ch=output_ch, base_ch=256, name="D_dst")
                    self.model_filename_list += [ [self.D_src, 'D_src.npy'] ]
                    self.model_filename_list += [ [self.D_dst, 'D_dst.npy'] ]

                # Initialize optimizers
                lr=5e-5
                clipnorm = 1.0 if self.options['clipgrad'] else 0.0
                self.src_dst_opt = nn.RMSprop(lr=lr, clipnorm=clipnorm, name='src_dst_opt')
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]
                
                self.src_dst_all_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights_ex(learn_mask) + self.decoder_dst.get_weights_ex(learn_mask)

                self.src_dst_opt.initialize_variables (self.src_dst_all_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu)

                if gan_power != 0:
                    self.D_src_dst_opt = nn.RMSprop(lr=lr, clipnorm=clipnorm, name='D_src_dst_opt')
                    self.D_src_dst_opt.initialize_variables ( self.D_src.get_weights()+self.D_dst.get_weights(), vars_on_cpu=optimizer_vars_on_cpu)
                    self.model_filename_list += [ (self.D_src_dst_opt, 'D_src_dst_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)


            # Compute losses per GPU
            gpu_pred_src_src_list = []
            gpu_pred_dst_dst_list = []
            gpu_pred_src_dst_list = []
            gpu_pred_src_srcm_list = []
            gpu_pred_dst_dstm_list = []
            gpu_pred_src_dstm_list = []

            gpu_src_losses = []
            gpu_dst_losses = []
            gpu_G_loss_gvs = []
            gpu_D_code_loss_gvs = []
            gpu_D_src_dst_loss_gvs = []
            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):

                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst      = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst      = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm_all = self.target_srcm_all[batch_slice,:,:,:]
                        gpu_target_dstm_all = self.target_dstm_all[batch_slice,:,:,:]
                        
                    # process model tensors                   
                    gpu_src_code     = self.inter(self.encoder(gpu_warped_src))
                    gpu_dst_code     = self.inter(self.encoder(gpu_warped_dst))
                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)
                    
                    # unpack masks from one combined mask
                    gpu_target_srcm      = tf.clip_by_value (gpu_target_srcm_all, 0, 1)                                   
                    gpu_target_dstm      = tf.clip_by_value (gpu_target_dstm_all, 0, 1)                    
                    gpu_target_srcm_eyes = tf.clip_by_value (gpu_target_srcm_all-1, 0, 1)     
                    gpu_target_dstm_eyes = tf.clip_by_value (gpu_target_dstm_all-1, 0, 1)
                    
                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt  = gpu_target_dst_masked if masked_training else gpu_target_dst
                    
                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    
                    if eyes_prio:
                        gpu_src_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_src*gpu_target_srcm_eyes - gpu_pred_src_src*gpu_target_srcm_eyes ), axis=[1,2,3])
                    
                    if learn_mask:
                        gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    gpu_dst_loss = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    
                    if eyes_prio:
                        gpu_dst_loss += tf.reduce_mean ( 300*tf.abs ( gpu_target_dst*gpu_target_dstm_eyes - gpu_pred_dst_dst*gpu_target_dstm_eyes ), axis=[1,2,3])
                    
                    if learn_mask:
                        gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss

                    def DLoss(labels,logits):
                        return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1,2,3])

                    if gan_power != 0:
                        gpu_pred_src_src_d       = self.D_src(gpu_pred_src_src_masked_opt)
                        gpu_pred_src_src_d_ones  = tf.ones_like (gpu_pred_src_src_d)
                        gpu_pred_src_src_d_zeros = tf.zeros_like(gpu_pred_src_src_d)
                        gpu_target_src_d         = self.D_src(gpu_target_src_masked_opt)
                        gpu_target_src_d_ones    = tf.ones_like(gpu_target_src_d)
                        gpu_pred_dst_dst_d       = self.D_dst(gpu_pred_dst_dst_masked_opt)
                        gpu_pred_dst_dst_d_ones  = tf.ones_like (gpu_pred_dst_dst_d)
                        gpu_pred_dst_dst_d_zeros = tf.zeros_like(gpu_pred_dst_dst_d)
                        gpu_target_dst_d         = self.D_dst(gpu_target_dst_masked_opt)
                        gpu_target_dst_d_ones    = tf.ones_like(gpu_target_dst_d)

                        gpu_D_src_dst_loss = (DLoss(gpu_target_src_d_ones   , gpu_target_src_d) + \
                                              DLoss(gpu_pred_src_src_d_zeros, gpu_pred_src_src_d) ) * 0.5 + \
                                             (DLoss(gpu_target_dst_d_ones   , gpu_target_dst_d) + \
                                              DLoss(gpu_pred_dst_dst_d_zeros, gpu_pred_dst_dst_d) ) * 0.5

                        gpu_D_src_dst_loss_gvs += [ nn.gradients (gpu_D_src_dst_loss, self.D_src.get_weights()+self.D_dst.get_weights() ) ]

                        gpu_G_loss += gan_power*(DLoss(gpu_pred_src_src_d_ones, gpu_pred_src_src_d) + DLoss(gpu_pred_dst_dst_d_ones, gpu_pred_dst_dst_d))


                    gpu_G_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                pred_src_src  = nn.concat(gpu_pred_src_src_list, 0)
                pred_dst_dst  = nn.concat(gpu_pred_dst_dst_list, 0)
                pred_src_dst  = nn.concat(gpu_pred_src_dst_list, 0)
                pred_src_srcm = nn.concat(gpu_pred_src_srcm_list, 0)
                pred_dst_dstm = nn.concat(gpu_pred_dst_dstm_list, 0)
                pred_src_dstm = nn.concat(gpu_pred_src_dstm_list, 0)
                src_loss = nn.average_tensor_list(gpu_src_losses)
                dst_loss = nn.average_tensor_list(gpu_dst_losses)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))

                if gan_power != 0:
                    src_D_src_dst_loss_gv_op = self.D_src_dst_opt.get_update_op (nn.average_gv_list(gpu_D_src_dst_loss_gvs) )


            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm_all, \
                              warped_dst, target_dst, target_dstm_all):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm_all:target_srcm_all,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm_all:target_dstm_all,
                                                       })
                s = np.mean(s)
                d = np.mean(d)
                return s, d
            self.src_dst_train = src_dst_train

            if gan_power != 0:
                def D_src_dst_train(warped_src, target_src, target_srcm_all, \
                                    warped_dst, target_dst, target_dstm_all):
                    nn.tf_sess.run ([src_D_src_dst_loss_gv_op], feed_dict={self.warped_src :warped_src,
                                                                           self.target_src :target_src,
                                                                           self.target_srcm_all:target_srcm_all,
                                                                           self.warped_dst :warped_dst,
                                                                           self.target_dst :target_dst,
                                                                           self.target_dstm_all:target_dstm_all})
                self.D_src_dst_train = D_src_dst_train

            if learn_mask:
                def AE_view(warped_src, warped_dst):
                    return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                             feed_dict={self.warped_src:warped_src,
                                                        self.warped_dst:warped_dst})
            else:
                def AE_view(warped_src, warped_dst):
                    return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_src_dst],
                                             feed_dict={self.warped_src:warped_src,
                                                        self.warped_dst:warped_dst})
            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            if learn_mask:
                def AE_merge( warped_dst):
                    return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})
            else:
                def AE_merge( warped_dst):
                    return nn.tf_sess.run ( [gpu_pred_src_dst], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            if self.pretrain_just_disabled:
                do_init = False
                if model == self.inter:
                    do_init = True
            else:
                do_init = self.is_first_run()

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_HEAD
                
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            t_img_warped = t.IMG_WARPED_TRANSFORMED

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2
    
            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types = [ {'types' : (t_img_warped, face_type, t.MODE_BGR),      'data_format':nn.data_format, 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'data_format':nn.data_format, 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_EYES_HULL), 'data_format':nn.data_format, 'resolution': resolution },
                                              ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types = [ {'types' : (t_img_warped, face_type, t.MODE_BGR),      'data_format':nn.data_format, 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'data_format':nn.data_format, 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_EYES_HULL), 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        generators_count=dst_generators_count )
                             ])

            if self.pretrain_just_disabled:
                self.update_sample_for_preview(force_new=True)

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )


    #override
    def onTrainOneIter(self):
        ( (warped_src, target_src, target_srcm_all), \
          (warped_dst, target_dst, target_dstm_all) ) = self.generate_next_samples()

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm_all, warped_dst, target_dst, target_dstm_all)

        if self.gan_power != 0:
            self.D_src_dst_train (warped_src, target_src, target_srcm_all, warped_dst, target_dst, target_dstm_all)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples):
        ( (warped_src, target_src, target_srcm_all,),
          (warped_dst, target_dst, target_dstm_all,) ) = samples

        if self.learn_mask:
            S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
            DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]
        else:
            S, D, SS, DD, SD, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format) , 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]

        target_srcm_all, target_dstm_all = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm_all, target_dstm_all] )]
        
        target_srcm = np.clip(target_srcm_all, 0, 1)
        target_dstm = np.clip(target_dstm_all, 0, 1)
        
        n_samples = min(4, self.get_batch_size(), 800 // self.resolution )

        result = []
        st = []
        for i in range(n_samples):
            ar = D[i], DD[i], SD[i]

            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('HEAD', np.concatenate (st, axis=0 )), ]

        if self.learn_mask:
            st_m = []
            for i in range(n_samples):
                ar = D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('HEAD masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        if self.learn_mask:
            bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

            mask = mask_dst_dstm[0] * mask_src_dstm[0]
            return bgr[0], mask[...,0]
        else:
            bgr, = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]
            return bgr[0]

    #override
    def get_MergerConfig(self):
        face_type = FaceType.HEAD
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigHead()

Model = HEADModel

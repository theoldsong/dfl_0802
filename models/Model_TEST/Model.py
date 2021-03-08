import multiprocessing
from functools import partial

import cv2
import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from core.cv2ex import *
from facelib import FaceType
from models import ModelBase
from samplelib import *


class TESTModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()

        lowest_vram = 2
        if len(device_config.devices) != 0:
            lowest_vram = device_config.devices.get_worst_device().total_mem_gb

        if lowest_vram >= 4:
            suggest_batch_size = 8
        else:
            suggest_batch_size = 4

        yn_str = {True:'y',False:'n'}
        ask_override = self.ask_override()

        resolution = default_resolution = self.options['resolution']         = self.load_or_def_option('resolution', 512)

        if self.is_first_run() or ask_override:
            self.ask_batch_size(suggest_batch_size)
            resolution = io.input_int("Resolution", default_resolution, add_info="64-1024")

        self.stage_max = stage_max = np.clip( mathlib.get_power_of_two(resolution), 6, 10 )-2
        self.options['resolution'] = resolution = 2**(stage_max+2)


        default_stage             = self.load_or_def_option('stage', 0)
        default_target_stage_iter = self.load_or_def_option('target_stage_iter', self.iter+100000 )

        if (self.is_first_run() or ask_override):
            new_stage = np.clip ( io.input_int("Stage", default_stage, add_info=f"0-{stage_max}"), 0, stage_max )
            if new_stage != default_stage:
                self.options['start_stage_iter'] = self.iter
                default_target_stage_iter = self.iter+100000
            self.options['stage'] = new_stage
        else:
            self.options['stage'] = default_stage

        if self.options['stage'] == 0:
            if 'start_stage_iter' in self.options:
                self.options.pop('start_stage_iter')

            if 'target_stage_iter' in self.options:
                self.options.pop('target_stage_iter')
        else:
            if (self.is_first_run() or ask_override):
                self.options['target_stage_iter'] = io.input_int("Target stage iteration", default_target_stage_iter)
            else:
                self.options['target_stage_iter'] = default_target_stage_iter


    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        class EncBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, level):
                self.zero_level = level == 0
                self.conv1 = nn.Conv2D(in_ch, out_ch, kernel_size=3, padding='SAME')
                self.conv2 = nn.Conv2D(out_ch, out_ch, kernel_size=4 if self.zero_level else 3, padding='VALID' if self.zero_level else 'SAME')

            def forward(self, x):
                x = tf.nn.leaky_relu( self.conv1(x), 0.2 )
                x = tf.nn.leaky_relu( self.conv2(x), 0.2 )

                if not self.zero_level:
                    x = nn.max_pool(x)
                    
                #if self.zero_level:
                    
                    
                return x

        class DecBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, level):
                self.zero_level = level == 0
                self.conv1 = nn.Conv2D(in_ch, out_ch, kernel_size=4 if self.zero_level else 3, padding=3 if self.zero_level else 'SAME')
                self.conv2 = nn.Conv2D(out_ch, out_ch, kernel_size=3, padding='SAME')


            def forward(self, x):
                if not self.zero_level:
                    x = nn.upsample2d(x)

                x = tf.nn.leaky_relu( self.conv1(x), 0.2 )
                x = tf.nn.leaky_relu( self.conv2(x), 0.2 )
                return x
        
        class InterBlock(nn.ModelBase):
            def on_build(self, in_ch, out_ch, level):
                self.zero_level = level == 0
                self.dense1 = nn.Dense()
                
            def forward(self, x):
                x = tf.nn.leaky_relu( self.conv1(x), 0.2 )
                x = tf.nn.leaky_relu( self.conv2(x), 0.2 )

                if not self.zero_level:
                    x = nn.max_pool(x)
                    
                #if self.zero_level:
                    
                    
                return x
                
        class FromRGB(nn.ModelBase):
            def on_build(self, out_ch):
                self.conv1 = nn.Conv2D(3, out_ch, kernel_size=1, padding='SAME')

            def forward(self, x):
                return tf.nn.leaky_relu( self.conv1(x), 0.2 )

        class ToRGB(nn.ModelBase):
            def on_build(self, in_ch):
                self.conv = nn.Conv2D(in_ch, 3, kernel_size=1, padding='SAME')
                self.convm = nn.Conv2D(in_ch, 1, kernel_size=1, padding='SAME')

            def forward(self, x):
                return tf.nn.sigmoid( self.conv(x) ), tf.nn.sigmoid( self.convm(x) )
                
        ed_dims = 16
        ae_res = 4
        level_chs = { i-1:v for i,v in enumerate([np.clip(ed_dims*(2**i), 0, 512) for i in range(self.stage_max+2)][::-1]) }
        ae_ch = level_chs[0]
        
        


        class Encoder(nn.ModelBase):
            def on_build(self, e_ch, levels):
                self.enc_blocks = {}
                self.from_rgbs = {}

                self.dense_norm = nn.DenseNorm()

                for level in range(levels,-1,-1):                    
                    self.from_rgbs[level] = FromRGB(level_chs[level])
                    if level != 0:
                        self.enc_blocks[level] = EncBlock(level_chs[level], level_chs[level-1], level)
                        
                self.ae_dense1 = nn.Dense ( ae_res*ae_res*ae_ch, 256 )
                self.ae_dense2 = nn.Dense ( 256, ae_res*ae_res*ae_ch )
                
            def forward(self, stage, inp, prev_inp=None, alpha=None):
                x = inp

                for level in range(stage, -1, -1):
                    if stage in self.from_rgbs:
                        if level == stage:
                            x = self.from_rgbs[level](x)
                        elif level == stage-1:
                            x = x*alpha + self.from_rgbs[level](prev_inp)*(1-alpha)
                            
                        if level != 0:
                            x = self.enc_blocks[level](x)

                x = nn.flatten(x)
                x = self.dense_norm(x)                
                x = self.ae_dense1(x)                
                x = self.ae_dense2(x)    
                x = nn.reshape_4D (x, ae_res, ae_res, ae_ch)
                   
                return x

            def get_stage_weights(self, stage):
                self.get_weights()
                weights = []
                for level in range(stage, -1, -1):
                    if stage in self.from_rgbs:
                        if level == stage or level == stage-1:
                            weights.append ( self.from_rgbs[level].get_weights() )                            
                        if level != 0:
                            weights.append ( self.enc_blocks[level].get_weights() )
                weights.append ( self.ae_dense1.get_weights() )
                weights.append ( self.ae_dense2.get_weights() )

                if len(weights) == 0:
                    return []
                elif len(weights) == 1:
                    return weights[0]
                else:
                    return sum(weights[1:],weights[0])

        class Decoder(nn.ModelBase):
            def on_build(self, levels_range):

                self.dec_blocks = {}
                self.to_rgbs = {}

                for level in range(levels_range[0],levels_range[1]+1):
                    self.to_rgbs[level] = ToRGB( level_chs[level] )
                    if level != 0:
                        self.dec_blocks[level] = DecBlock(level_chs[level-1], level_chs[level], level)
                    
                
                    
            def forward(self, stage, inp, alpha=None, inter=None):
                x = inp
              
                for level in range(stage+1):
                    if level in self.to_rgbs:
                        
                        if level == stage and stage > 0:
                            prev_level = level-1
                            #prev_x, prev_xm = (inter.to_rgbs[prev_level] if inter is not None and prev_level in inter.to_rgbs else self.to_rgbs[prev_level])(x)
                            prev_x, prev_xm = self.to_rgbs[prev_level](x)
                            
                            prev_x = nn.upsample2d(prev_x)
                            prev_xm = nn.upsample2d(prev_xm)
                            
                        if level != 0:
                            x = self.dec_blocks[level](x)
                        
                        if level == stage:
                            x,xm = self.to_rgbs[level](x)
                            if stage > 0:
                                x = x*alpha + prev_x*(1-alpha)
                                xm = xm*alpha + prev_xm*(1-alpha)
                            return x, xm
                return x
                        
            def get_stage_weights(self, stage):
                # Call internal get_weights in order to initialize inner logic
                self.get_weights()

                weights = []
                for level in range(stage+1):
                    if level in self.to_rgbs:
                        if level != 0:
                            weights.append ( self.dec_blocks[level].get_weights() )                        
                        if level == stage or level == stage-1:
                            weights.append ( self.to_rgbs[level].get_weights() )

                if len(weights) == 0:
                    return []
                elif len(weights) == 1:
                    return weights[0]
                else:
                    return sum(weights[1:],weights[0])




        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        self.stage = stage = self.options['stage']
        self.start_stage_iter = self.options.get('start_stage_iter', 0)
        self.target_stage_iter = self.options.get('target_stage_iter', 0)

        resolution = self.resolution = self.options['resolution']
        stage_resolutions = [ 2**(i+2) for i in range(self.stage_max+1) ]
        stage_resolution = self.stage_resolution = stage_resolutions[stage]
        prev_stage = stage-1 if stage != 0 else stage
        prev_stage_resolution = stage_resolutions[stage-1] if stage != 0 else stage_resolution

        

        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        models_opt_on_gpu = len(devices) == 1 and devices[0].total_mem_gb >= 4
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_nc = 3
        output_nc = 3
        prev_bgr_shape = nn.get4Dshape(prev_stage_resolution, prev_stage_resolution, output_nc)
        bgr_shape = nn.get4Dshape(stage_resolution, stage_resolution, output_nc)
        mask_shape = nn.get4Dshape(stage_resolution, stage_resolution, 1)

        self.model_filename_list = []

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.prev_warped_src = tf.placeholder (tf.float32, prev_bgr_shape)
            self.prev_warped_dst = tf.placeholder (tf.float32, prev_bgr_shape)
            
            self.warped_src = tf.placeholder (tf.float32, bgr_shape)
            self.warped_dst = tf.placeholder (tf.float32, bgr_shape)

            self.target_src = tf.placeholder (tf.float32, bgr_shape)
            self.target_dst = tf.placeholder (tf.float32, bgr_shape)

            self.target_srcm = tf.placeholder (tf.float32, mask_shape)
            self.target_dstm = tf.placeholder (tf.float32, mask_shape)
            self.alpha_t = tf.placeholder (tf.float32, (None,1,1,1) )
            """
            self.stage_warped_src = nn.resize2d_nearest(self.warped_src, -2**(self.stage_max-stage))
            self.stage_warped_dst = nn.resize2d_nearest(self.warped_dst, -2**(self.stage_max-stage))
            self.prev_stage_warped_src = nn.resize2d_nearest(self.warped_src, -2**(self.stage_max-stage+1) ) if stage > 0 else self.stage_warped_src
            self.prev_stage_warped_dst = nn.resize2d_nearest(self.warped_dst, -2**(self.stage_max-stage+1) ) if stage > 0 else self.stage_warped_dst
            self.stage_target_src = nn.resize2d_nearest(self.target_src, -2**(self.stage_max-stage))
            self.stage_target_dst = nn.resize2d_nearest(self.target_dst, -2**(self.stage_max-stage))
            self.stage_target_srcm = nn.resize2d_nearest(self.target_srcm, -2**(self.stage_max-stage))
            self.stage_target_dstm = nn.resize2d_nearest(self.target_dstm, -2**(self.stage_max-stage))
            """
            
            #import code
            #code.interact(local=dict(globals(), **locals()))


        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(e_ch=ed_dims, levels=self.stage_max, name='encoder')



            #self.inter = Decoder(d_ch=ed_dims, total_levels=self.stage_max, levels_range=[0,2], name='inter')
            self.decoder_src = Decoder(levels_range=[0,self.stage_max], name='decoder_src')
            self.decoder_dst = Decoder(levels_range=[0,self.stage_max], name='decoder_dst')



            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          #[self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                self.src_dst_all_weights = self.encoder.get_weights() +  self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                self.src_dst_trainable_weights = self.encoder.get_stage_weights(stage) \
                               + self.decoder_src.get_stage_weights(stage) \
                               + self.decoder_dst.get_stage_weights(stage)

                # Initialize optimizers
                self.src_dst_opt = nn.RMSprop(lr=2e-4, lr_dropout=0.3, name='src_dst_opt')
                self.src_dst_opt.initialize_variables(self.src_dst_all_weights, vars_on_cpu=optimizer_vars_on_cpu )
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

        
        
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
            gpu_src_dst_loss_gvs = []

            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        """
                        gpu_stage_warped_src   = self.stage_warped_src [batch_slice,:,:,:]
                        gpu_stage_warped_dst   = self.stage_warped_dst [batch_slice,:,:,:]
                        gpu_prev_stage_warped_src = self.prev_stage_warped_src[batch_slice,:,:,:]
                        gpu_prev_stage_warped_dst = self.prev_stage_warped_dst[batch_slice,:,:,:]
                        gpu_stage_target_src   = self.stage_target_src [batch_slice,:,:,:]
                        gpu_stage_target_dst   = self.stage_target_dst [batch_slice,:,:,:]
                        gpu_stage_target_srcm  = self.stage_target_srcm[batch_slice,:,:,:]
                        gpu_stage_target_dstm  = self.stage_target_dstm[batch_slice,:,:,:]                    
                        gpu_alpha_t      = self.alpha_t[batch_slice,:,:,:]
                        """
                        gpu_prev_warped_src   = self.prev_warped_src [batch_slice,:,:,:]
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_prev_warped_dst   = self.prev_warped_dst [batch_slice,:,:,:]
                        gpu_warped_dst   = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm  = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_dstm  = self.target_dstm[batch_slice,:,:,:]                    
                        gpu_alpha_t      = self.alpha_t[batch_slice,:,:,:]
                        
                    
                    # process model tensors
                    #gpu_src_code     = self.inter(stage, self.encoder(stage, gpu_warped_src, gpu_prev_warped_src, gpu_alpha_t), gpu_alpha_t )
                    #gpu_dst_code     = self.inter(stage, self.encoder(stage, gpu_warped_dst, gpu_prev_warped_dst, gpu_alpha_t), gpu_alpha_t )

                    gpu_src_code     = self.encoder(stage, gpu_warped_src, gpu_prev_warped_src, gpu_alpha_t)
                    gpu_dst_code     = self.encoder(stage, gpu_warped_dst, gpu_prev_warped_src, gpu_alpha_t)


                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(stage, gpu_src_code, gpu_alpha_t)#, inter=self.inter)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(stage, gpu_dst_code, gpu_alpha_t)#, inter=self.inter)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(stage, gpu_dst_code, gpu_alpha_t)#, inter=self.inter)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, stage_resolution // 32) )
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, stage_resolution // 32) )

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_srcmasked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss = tf.reduce_mean ( 10*tf.square ( gpu_target_srcmasked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( tf.square( gpu_target_srcm - gpu_pred_src_srcm ), axis=[1,2,3] )
                    if stage_resolution >= 16:
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(stage_resolution/11.6)), axis=[1])
                    if stage_resolution >= 32:
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(stage_resolution/23.2)), axis=[1])
                    
                    gpu_dst_loss = tf.reduce_mean ( 10*tf.square( gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )
                    if stage_resolution >= 16:
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(stage_resolution/11.6) ), axis=[1])
                    if stage_resolution >= 32:
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(stage_resolution/23.2) ), axis=[1])
                    
                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_src_dst_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [ nn.gradients ( gpu_src_dst_loss, self.src_dst_trainable_weights ) ]
                    """
                    gpu_src_code     = self.encoder(stage, gpu_stage_warped_src, gpu_prev_stage_warped_src, gpu_alpha_t)
                    gpu_dst_code     = self.encoder(stage, gpu_stage_warped_dst, gpu_prev_stage_warped_src, gpu_alpha_t)


                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(stage, gpu_src_code, gpu_alpha_t)#, inter=self.inter)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(stage, gpu_dst_code, gpu_alpha_t)#, inter=self.inter)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(stage, gpu_dst_code, gpu_alpha_t)#, inter=self.inter)

                    gpu_pred_src_src_list.append(gpu_pred_src_src)
                    gpu_pred_dst_dst_list.append(gpu_pred_dst_dst)
                    gpu_pred_src_dst_list.append(gpu_pred_src_dst)

                    gpu_pred_src_srcm_list.append(gpu_pred_src_srcm)
                    gpu_pred_dst_dstm_list.append(gpu_pred_dst_dstm)
                    gpu_pred_src_dstm_list.append(gpu_pred_src_dstm)

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_stage_target_srcm,  max(1, stage_resolution // 32) )
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_stage_target_dstm,  max(1, stage_resolution // 32) )

                    gpu_target_dst_masked      = gpu_stage_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_stage_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_srcmasked_opt  = gpu_stage_target_src*gpu_target_srcm_blur if masked_training else gpu_stage_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_stage_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss = tf.reduce_mean ( 10*tf.square ( gpu_target_srcmasked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( tf.square( gpu_stage_target_srcm - gpu_pred_src_srcm ), axis=[1,2,3] )
                    if stage_resolution >= 16:
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(stage_resolution/11.6)), axis=[1])
                    if stage_resolution >= 32:
                        gpu_src_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(stage_resolution/23.2)), axis=[1])
                    
                    gpu_dst_loss = tf.reduce_mean ( 10*tf.square( gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( tf.square( gpu_stage_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )
                    if stage_resolution >= 16:
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(stage_resolution/11.6) ), axis=[1])
                    if stage_resolution >= 32:
                        gpu_dst_loss += tf.reduce_mean ( 5*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(stage_resolution/23.2) ), axis=[1])
                    
                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_src_dst_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [ nn.gradients ( gpu_src_dst_loss, self.src_dst_trainable_weights ) ]
                    """

            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                if gpu_count == 1:
                    pred_src_src = gpu_pred_src_src_list[0]
                    pred_dst_dst = gpu_pred_dst_dst_list[0]
                    pred_src_dst = gpu_pred_src_dst_list[0]
                    pred_src_srcm = gpu_pred_src_srcm_list[0]
                    pred_dst_dstm = gpu_pred_dst_dstm_list[0]
                    pred_src_dstm = gpu_pred_src_dstm_list[0]

                    src_loss = gpu_src_losses[0]
                    dst_loss = gpu_dst_losses[0]
                    src_dst_loss_gv = gpu_src_dst_loss_gvs[0]
                else:
                    pred_src_src = tf.concat(gpu_pred_src_src_list, 0)
                    pred_dst_dst = tf.concat(gpu_pred_dst_dst_list, 0)
                    pred_src_dst = tf.concat(gpu_pred_src_dst_list, 0)
                    pred_src_srcm = tf.concat(gpu_pred_src_srcm_list, 0)
                    pred_dst_dstm = tf.concat(gpu_pred_dst_dstm_list, 0)
                    pred_src_dstm = tf.concat(gpu_pred_src_dstm_list, 0)

                    src_loss = nn.average_tensor_list(gpu_src_losses)
                    dst_loss = nn.average_tensor_list(gpu_dst_losses)
                    src_dst_loss_gv = nn.average_gv_list (gpu_src_dst_loss_gvs)

                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (src_dst_loss_gv)

            # Initializing training and view functions
            def get_alpha(batch_size):
                alpha = 0
                if self.stage != 0:
                    alpha = (self.iter - self.start_stage_iter) / ( self.target_stage_iter - self.start_stage_iter )
                    alpha = np.clip(alpha, 0, 1)
                alpha = np.array([alpha], nn.floatx.as_numpy_dtype).reshape( (1,1,1,1) )
                alpha = np.repeat(alpha, batch_size, 0)
                return alpha
                
            def src_dst_train(prev_warped_src, warped_src, target_src, target_srcm, \
                              prev_warped_dst, warped_dst, target_dst, target_dstm):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.prev_warped_src :prev_warped_src,
                                                       self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.prev_warped_dst :prev_warped_dst,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       self.alpha_t:get_alpha(prev_warped_src.shape[0])
                                                       })
                s = np.mean(s)
                d = np.mean(d)
                return s, d
            self.src_dst_train = src_dst_train

            def AE_view(prev_warped_src, warped_src, prev_warped_dst, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.prev_warped_src :prev_warped_src,
                                                       self.warped_src:warped_src,
                                                       self.prev_warped_dst :prev_warped_dst,
                                                       self.warped_dst:warped_dst,
                                                       self.alpha_t:get_alpha(prev_warped_src.shape[0]) 
                                                       })

            self.AE_view = AE_view
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code, stage=stage)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code, stage=stage)

            def AE_merge( warped_dst):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge




        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()

            if self.pretrain_just_disabled:
                if model == self.inter:
                    do_init = True

            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )

            if do_init:
                model.init_weights()

        # initializing sample generators

        if self.is_training:
            self.face_type = FaceType.FULL

            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = multiprocessing.cpu_count()

            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count - src_generators_count

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,  'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': prev_stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,  'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': prev_stage_resolution},                                                
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                              ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,  'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': prev_stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,  'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': prev_stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_MASK, 'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.G,   'face_mask_type' : SampleProcessor.FaceMaskType.FULL_FACE, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': stage_resolution},
                                              ],
                        generators_count=dst_generators_count )
                             ])


            self.last_samples = None

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )

    #override
    def onTrainOneIter(self):
        samples = self.last_samples = self.generate_next_samples()
        ( (prev_warped_src, warped_src, prev_target_src, target_src, target_srcm), \
          (prev_warped_dst, warped_dst, prev_target_dst, target_dst, target_dstm) ) = samples

        src_loss, dst_loss = self.src_dst_train (prev_warped_src, warped_src, target_src, target_srcm,
                                                 prev_warped_dst, warped_dst, target_dst, target_dstm)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples):
        n_samples = min(4 if self.stage_resolution < 256 else 1, self.get_batch_size() )

        ( (prev_warped_src, warped_src, prev_target_src, target_src, target_srcm),
          (prev_warped_dst, warped_dst, prev_target_dst, target_dst, target_dstm) ) = \
                [ [sample[0:n_samples] for sample in sample_list ]
                                                 for sample_list in samples ]

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (prev_target_src, target_src, prev_target_dst, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]
        
        target_srcm, target_dstm = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_srcm, target_dstm]) ]
        

        
        result = []
        
        if self.stage_resolution < 256:
            S, D, target_srcm, target_dstm = [ [cv2_resize(y, (self.stage_resolution, self.stage_resolution) ) for y in x] for x in ([S, D, target_srcm, target_dstm ]) ]
        
        
            st = []
            for i in range(n_samples):
                ar = S[i], SS[i], D[i], DD[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )

            result += [ ('test', np.concatenate (st, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
                st_m.append ( np.concatenate ( ar, axis=1) )

            result += [ ('test masked', np.concatenate (st_m, axis=0 )), ]
        else:
            SS, DD, DDM, SD, SDM = [ [cv2.resize(y, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST ) for y in x] for x in ([SS, DD, DDM, SD, SDM]) ]
        
            st = []
            for i in range(n_samples):
                ar = S[i], SS[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD src-src', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], DD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD dst-dst', np.concatenate (st, axis=0 )), ]

            st = []
            for i in range(n_samples):
                ar = D[i], SD[i]
                st.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD pred', np.concatenate (st, axis=0 )), ]


            st_m = []
            for i in range(n_samples):
                ar = S[i]*target_srcm[i], SS[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked src-src', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                ar = D[i]*target_dstm[i], DD[i]*DDM[i]
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked dst-dst', np.concatenate (st_m, axis=0 )), ]

            st_m = []
            for i in range(n_samples):
                SD_mask = DDM[i]*SDM[i] if self.face_type < FaceType.HEAD else SDM[i]
                ar = D[i]*target_dstm[i], SD[i]*SD_mask
                st_m.append ( np.concatenate ( ar, axis=1) )
            result += [ ('SAEHD masked pred', np.concatenate (st_m, axis=0 )), ]
            
        """
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('test', np.concatenate (st, axis=0 )), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('test masked', np.concatenate (st_m, axis=0 )), ]
        """

        return result

    def predictor_func (self, face=None):

        bgr, mask_dst_dstm, mask_src_dstm = self.AE_merge (face[np.newaxis,...])
        mask = mask_dst_dstm[0] * mask_src_dstm[0]
        return bgr[0], mask[...,0]

    #override
    def get_MergerConfig(self):
        face_type = FaceType.FULL

        import merger
        return self.predictor_func, (self.stage_resolution, self.stage_resolution, 3), merger.MergerConfigMasked(face_type=face_type,
                                     default_mode = 'overlay',
                                    )

Model = TESTModel

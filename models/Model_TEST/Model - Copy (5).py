import colorsys
import multiprocessing
import operator
from functools import partial
import cv2
import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *


class SAEHDModel(ModelBase):

    #override
    def on_initialize_options(self):
        device_config = nn.getCurrentDeviceConfig()


        yn_str = {True:'y',False:'n'}

        default_resolution         = self.options['resolution']         = self.load_or_def_option('resolution', 128)
        default_face_type          = self.options['face_type']          = self.load_or_def_option('face_type', 'f')
        #default_models_opt_on_gpu  = self.options['models_opt_on_gpu']  = self.load_or_def_option('models_opt_on_gpu', True)
        #default_archi              = self.options['archi']              = self.load_or_def_option('archi', 'df')
        #default_ae_dims            = self.options['ae_dims']            = self.load_or_def_option('ae_dims', 256)
        #default_e_dims             = self.options['e_dims']             = self.load_or_def_option('e_dims', 64)
        #default_d_dims             = self.options['d_dims']             = self.options.get('d_dims', None)
        #default_d_mask_dims        = self.options['d_mask_dims']        = self.options.get('d_mask_dims', None)
        #default_masked_training    = self.options['masked_training']    = self.load_or_def_option('masked_training', True)
        #default_eyes_prio          = self.options['eyes_prio']          = self.load_or_def_option('eyes_prio', False)
        #default_lr_dropout         = self.options['lr_dropout']         = self.load_or_def_option('lr_dropout', False)
        #default_random_warp        = self.options['random_warp']        = self.load_or_def_option('random_warp', True)
        #default_gan_power          = self.options['gan_power']          = self.load_or_def_option('gan_power', 0.0)
        #default_true_face_power    = self.options['true_face_power']    = self.load_or_def_option('true_face_power', 0.0)
        #default_face_style_power   = self.options['face_style_power']   = self.load_or_def_option('face_style_power', 0.0)
        #default_bg_style_power     = self.options['bg_style_power']     = self.load_or_def_option('bg_style_power', 0.0)
        #default_ct_mode            = self.options['ct_mode']            = self.load_or_def_option('ct_mode', 'none')
        #default_clipgrad           = self.options['clipgrad']           = self.load_or_def_option('clipgrad', False)
        #default_pretrain           = self.options['pretrain']           = self.load_or_def_option('pretrain', False)

        ask_override = self.ask_override()
        if self.is_first_run() or ask_override:
            #self.ask_autobackup_hour()
            #self.ask_write_preview_history()
            #self.ask_target_iter()
            #self.ask_random_flip()
            self.ask_batch_size(4)

        if self.is_first_run():
            resolution = io.input_int("Resolution", default_resolution, add_info="64-512", help_message="More resolution requires more VRAM and time to train. Value will be adjusted to multiple of 16.")
            resolution = np.clip ( (resolution // 16) * 16, 64, 512)
            self.options['resolution'] = resolution
            self.options['face_type'] = io.input_str ("Face type", default_face_type, ['h','mf','f','wf','head'], help_message="Half / mid face / full face / whole face / head. Half face has better resolution, but covers less area of cheeks. Mid face is 30% wider than half face. 'Whole face' covers full area of face include forehead. 'head' covers full head, but requires XSeg for src and dst faceset.").lower()
            #self.options['archi'] = io.input_str ("AE architecture", default_archi, ['df','liae','dfhd','liaehd','dfuhd','liaeuhd'], help_message="'df' keeps faces more natural.\n'liae' can fix overly different face shapes.\n'hd' are experimental versions.").lower()

        return
        default_d_dims             = 48 if self.options['archi'] == 'dfhd' else 64
        default_d_dims             = self.options['d_dims']             = self.load_or_def_option('d_dims', default_d_dims)

        default_d_mask_dims        = default_d_dims // 3
        default_d_mask_dims        += default_d_mask_dims % 2
        default_d_mask_dims        = self.options['d_mask_dims']        = self.load_or_def_option('d_mask_dims', default_d_mask_dims)

        if self.is_first_run():
            self.options['ae_dims'] = np.clip ( io.input_int("AutoEncoder dimensions", default_ae_dims, add_info="32-1024", help_message="All face information will packed to AE dims. If amount of AE dims are not enough, then for example closed eyes will not be recognized. More dims are better, but require more VRAM. You can fine-tune model size to fit your GPU." ), 32, 1024 )

            e_dims = np.clip ( io.input_int("Encoder dimensions", default_e_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['e_dims'] = e_dims + e_dims % 2


            d_dims = np.clip ( io.input_int("Decoder dimensions", default_d_dims, add_info="16-256", help_message="More dims help to recognize more facial features and achieve sharper result, but require more VRAM. You can fine-tune model size to fit your GPU." ), 16, 256 )
            self.options['d_dims'] = d_dims + d_dims % 2

            d_mask_dims = np.clip ( io.input_int("Decoder mask dimensions", default_d_mask_dims, add_info="16-256", help_message="Typical mask dimensions = decoder dimensions / 3. If you manually cut out obstacles from the dst mask, you can increase this parameter to achieve better quality." ), 16, 256 )
            self.options['d_mask_dims'] = d_mask_dims + d_mask_dims % 2

        if self.is_first_run() or ask_override:
            if self.options['face_type'] == 'wf' or self.options['face_type'] == 'head':
                self.options['masked_training']  = io.input_bool ("Masked training", default_masked_training, help_message="This option is available only for 'whole_face' type. Masked training clips training area to full_face mask, thus network will train the faces properly.  When the face is trained enough, disable this option to train all area of the frame. Merge with 'raw-rgb' mode, then use Adobe After Effects to manually mask and compose whole face include forehead.")

            self.options['eyes_prio']  = io.input_bool ("Eyes priority", default_eyes_prio, help_message='Helps to fix eye problems during training like "alien eyes" and wrong eyes direction ( especially on HD architectures ) by forcing the neural network to train eyes with higher priority. before/after https://i.imgur.com/YQHOuSR.jpg ')

        if self.is_first_run() or ask_override:
            self.options['models_opt_on_gpu'] = io.input_bool ("Place models and optimizer on GPU", default_models_opt_on_gpu, help_message="When you train on one GPU, by default model and optimizer weights are placed on GPU to accelerate the process. You can place they on CPU to free up extra VRAM, thus set bigger dimensions.")

            self.options['lr_dropout']  = io.input_bool ("Use learning rate dropout", default_lr_dropout, help_message="When the face is trained enough, you can enable this option to get extra sharpness and reduce subpixel shake for less amount of iterations.")
            self.options['random_warp'] = io.input_bool ("Enable random warp of samples", default_random_warp, help_message="Random warp is required to generalize facial expressions of both faces. When the face is trained enough, you can disable it to get extra sharpness and reduce subpixel shake for less amount of iterations.")

            self.options['gan_power'] = np.clip ( io.input_number ("GAN power", default_gan_power, add_info="0.0 .. 10.0", help_message="Train the network in Generative Adversarial manner. Accelerates the speed of training. Forces the neural network to learn small details of the face. You can enable/disable this option at any time. Typical value is 1.0"), 0.0, 10.0 )

            if 'df' in self.options['archi']:
                self.options['true_face_power'] = np.clip ( io.input_number ("'True face' power.", default_true_face_power, add_info="0.0000 .. 1.0", help_message="Experimental option. Discriminates result face to be more like src face. Higher value - stronger discrimination. Typical value is 0.01 . Comparison - https://i.imgur.com/czScS9q.png"), 0.0, 1.0 )
            else:
                self.options['true_face_power'] = 0.0

            self.options['face_style_power'] = np.clip ( io.input_number("Face style power", default_face_style_power, add_info="0.0..100.0", help_message="Learn the color of the predicted face to be the same as dst inside mask. If you want to use this option with 'whole_face' you have to use XSeg trained mask. Warning: Enable it only after 10k iters, when predicted face is clear enough to start learn style. Start from 0.001 value and check history changes. Enabling this option increases the chance of model collapse."), 0.0, 100.0 )
            self.options['bg_style_power'] = np.clip ( io.input_number("Background style power", default_bg_style_power, add_info="0.0..100.0", help_message="Learn the area outside mask of the predicted face to be the same as dst. If you want to use this option with 'whole_face' you have to use XSeg trained mask. For whole_face you have to use XSeg trained mask. This can make face more like dst. Enabling this option increases the chance of model collapse. Typical value is 2.0"), 0.0, 100.0 )

            self.options['ct_mode'] = io.input_str (f"Color transfer for src faceset", default_ct_mode, ['none','rct','lct','mkl','idt','sot'], help_message="Change color distribution of src samples close to dst samples. Try all modes to find the best.")
            self.options['clipgrad'] = io.input_bool ("Enable gradient clipping", default_clipgrad, help_message="Gradient clipping reduces chance of model collapse, sacrificing speed of training.")

            self.options['pretrain'] = io.input_bool ("Enable pretraining mode", default_pretrain, help_message="Pretrain the model with large amount of various faces. After that, model can be used to train the fakes more quickly.")

        if self.options['pretrain'] and self.get_pretraining_data_path() is None:
            raise Exception("pretraining_data_path is not defined")

        self.pretrain_just_disabled = (default_pretrain == True and self.options['pretrain'] == False)

    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NHWC"#"NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        self.resolution = resolution = self.options['resolution']
        self.face_type = {'h'  : FaceType.HALF,
                          'mf' : FaceType.MID_FULL,
                          'f'  : FaceType.FULL,
                          'wf' : FaceType.WHOLE_FACE,
                          'head' : FaceType.HEAD}[ self.options['face_type'] ]


        models_opt_on_gpu = True#False if len(devices) == 0 else self.options['models_opt_on_gpu']
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch=3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)
        self.model_filename_list = []

        class BaseModel(nn.ModelBase):
            def on_build(self, in_ch, base_ch, out_ch=None):
                self.convs = [ nn.Conv2D( in_ch, base_ch, kernel_size=7, strides=1, padding='SAME'),
                               nn.Conv2D( base_ch, base_ch, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch, base_ch*2, kernel_size=3, strides=2, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*2, base_ch*2, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch*2, base_ch*4, kernel_size=3, strides=2, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*4, base_ch*4, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch*4, base_ch*8, kernel_size=3, strides=2, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*8, base_ch*8, kernel_size=3, strides=1, use_bias=False, padding='SAME')
                             ]

                self.frns = [ None,
                              nn.FRNorm2D(base_ch),
                              nn.FRNorm2D(base_ch*2),
                              nn.FRNorm2D(base_ch*2),
                              nn.FRNorm2D(base_ch*4),
                              nn.FRNorm2D(base_ch*4),
                              nn.FRNorm2D(base_ch*8),
                              nn.FRNorm2D(base_ch*8),
                            ]

                self.tlus = [ nn.TLU(base_ch),
                              nn.TLU(base_ch),
                              nn.TLU(base_ch*2),
                              nn.TLU(base_ch*2),
                              nn.TLU(base_ch*4),
                              nn.TLU(base_ch*4),
                              nn.TLU(base_ch*8),
                              nn.TLU(base_ch*8),
                            ]

                if out_ch is not None:
                    self.out_conv = nn.Conv2D( base_ch*8, out_ch, kernel_size=1, strides=1,  use_bias=False, padding='VALID')
                else:
                    self.out_conv = None

            def forward(self, inp):
                x = inp

                for i in range(len(self.convs)):
                    x = self.convs[i](x)
                    if self.frns[i] is not None:
                        x = self.frns[i](x)
                    x = self.tlus[i](x)

                if self.out_conv is not None:
                    x = self.out_conv(x)
                return x

        class Regressor(nn.ModelBase):
            def on_build(self, lmrks_ch, base_ch, out_ch):
                self.convs = [ nn.Conv2D( base_ch*8+lmrks_ch, base_ch*8, kernel_size=3, strides=1, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*8, base_ch*8*4, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch*8, base_ch*4, kernel_size=3, strides=1, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*4, base_ch*4*4, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch*4, base_ch*2, kernel_size=3, strides=1, use_bias=False, padding='SAME'),
                               nn.Conv2D( base_ch*2, base_ch*2*4, kernel_size=3, strides=1, use_bias=False, padding='SAME'),

                               nn.Conv2D( base_ch*2, base_ch, kernel_size=3, strides=1, use_bias=False, padding='SAME'),
                             ]

                self.frns = [ nn.FRNorm2D(base_ch*8),
                              nn.FRNorm2D(base_ch*8*4),
                              nn.FRNorm2D(base_ch*4),
                              nn.FRNorm2D(base_ch*4*4),
                              nn.FRNorm2D(base_ch*2),
                              nn.FRNorm2D(base_ch*2*4),
                              nn.FRNorm2D(base_ch),
                            ]

                self.tlus = [ nn.TLU(base_ch*8),
                              nn.TLU(base_ch*8*4),
                              nn.TLU(base_ch*4),
                              nn.TLU(base_ch*4*4),
                              nn.TLU(base_ch*2),
                              nn.TLU(base_ch*2*4),
                              nn.TLU(base_ch),
                            ]

                self.use_upscale = [ False,
                                    True,
                                    False,
                                    True,
                                    False,
                                    True,
                                    False,
                                  ]

                self.out_conv = nn.Conv2D( base_ch, out_ch, kernel_size=3, strides=1, padding='SAME')

            def forward(self, inp):
                x = inp

                for i in range(len(self.convs)):
                    x = self.convs[i](x)
                    x = self.frns[i](x)
                    x = self.tlus[i](x)

                    if self.use_upscale[i]:
                        x = nn.depth_to_space(x, 2)

                x = self.out_conv(x)
                x = tf.nn.sigmoid(x)
                return x

        def get_coord(x, other_axis, axis_size):
            # get "x-y" coordinates:
            g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
            g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
            coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
            coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
            g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
            return g_c, g_c_prob

        def get_gaussian_maps(mu_x, mu_y, width, height, inv_std=10.0, mode='rot'):
            """
            Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
            given the gaussian centers: MU [B, NMAPS, 2] tensor.
            STD: is the fixed standard dev.
            """
            y = tf.to_float(tf.linspace(-1.0, 1.0, width))
            x = tf.to_float(tf.linspace(-1.0, 1.0, height))

            if mode in ['rot', 'flat']:
                mu_y, mu_x = mu_y[...,None,None], mu_x[...,None,None]

                y = tf.reshape(y, [1, 1, width, 1])
                x = tf.reshape(x, [1, 1, 1, height])

                g_y = tf.square(y - mu_y)
                g_x = tf.square(x - mu_x)
                dist = (g_y + g_x) * inv_std**2

                if mode == 'rot':
                    g_yx = tf.exp(-dist)
                else:
                    g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

            elif mode == 'ankush':
                y = tf.reshape(y, [1, 1, width])
                x = tf.reshape(x, [1, 1, height])


                g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y[...,None] - y) * inv_std)))
                g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x[...,None] - x) * inv_std)))

                g_y = tf.expand_dims(g_y, axis=3)
                g_x = tf.expand_dims(g_x, axis=2)
                g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

            else:
                raise ValueError('Unknown mode: ' + str(mode))

            g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
            return g_yx

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_src = tf.placeholder (nn.floatx, bgr_shape)



        # Initializing model classes
        #model_archi = nn.DeepFakeArchi(resolution, mod='uhd' if 'uhd' in archi else None)
        self.landmarks_count = 512
        self.n_ch = 32
        with tf.device (models_opt_device):
            self.detector = BaseModel(3, self.n_ch, out_ch=self.landmarks_count, name='Detector')
            self.extractor = BaseModel(3, self.n_ch, name='Extractor')
            self.regressor = Regressor(self.landmarks_count, self.n_ch, 3, name='Regressor')



            self.model_filename_list += [ [self.detector,  'detector.npy'],
                                          [self.extractor, 'extractor.npy'],
                                          [self.regressor, 'regressor.npy'] ]

            if self.is_training:
                 # Initialize optimizers
                lr=5e-5
                lr_dropout = 0.3#0.3 if self.options['lr_dropout'] and not self.pretrain else 1.0
                clipnorm = 0.0#1.0 if self.options['clipgrad'] else 0.0
                self.model_opt = nn.RMSprop(lr=lr, lr_dropout=lr_dropout, clipnorm=clipnorm, name='model_opt')
                self.model_filename_list += [ (self.model_opt, 'model_opt.npy') ]

                self.model_trainable_weights = self.detector.get_weights() + self.extractor.get_weights() + self.regressor.get_weights()
                self.model_opt.initialize_variables (self.model_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu)



        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, self.get_batch_size() // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # Compute losses per GPU
            gpu_src_rec_list = []
            gauss_mu_list = []
            
            gpu_src_losses = []
            gpu_G_loss_gvs = []
            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):

                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                        gpu_warped_src      = self.warped_src [batch_slice,:,:,:]
                        gpu_target_src      = self.target_src [batch_slice,:,:,:]

                    # process model tensors

                    gpu_src_feat     = self.extractor(gpu_warped_src)
                    gpu_src_heatmaps = self.detector(gpu_target_src)

                    gauss_y, gauss_y_prob = get_coord(gpu_src_heatmaps, 2, gpu_src_heatmaps.shape.as_list()[1] )
                    gauss_x, gauss_x_prob = get_coord(gpu_src_heatmaps, 1, gpu_src_heatmaps.shape.as_list()[2] )
                    gauss_mu = tf.stack ( (gauss_x, gauss_y), -1)
                    
                    dist_loss = []
                    for i in range(self.landmarks_count):
                        
                        t = tf.concat( (gauss_mu[:,0:i], gauss_mu[:,i+1:] ), axis=1 )
                        
                        
                        diff = t - gauss_mu[:,i:i+1]
                        dist = tf.sqrt( diff[...,0]**2+diff[...,1]**2 )
                        
                        dist_loss += [ tf.reduce_mean(2.0 - dist,-1)  ]
                        
                    dist_loss = sum(dist_loss) / self.landmarks_count
                    #import code
                    #code.interact(local=dict(globals(), **locals()))

                    
                    
                    gauss_xy = get_gaussian_maps ( gauss_x, gauss_y, 16, 16 )

                    gpu_src_rec = self.regressor( tf.concat ( (gpu_src_feat, gauss_xy), -1) )

                    gpu_src_rec_list.append(gpu_src_rec)
                    gauss_mu_list.append(gauss_mu)
                    
                    gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src, gpu_src_rec, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square (gpu_target_src - gpu_src_rec), axis=[1,2,3])
                    gpu_src_loss += dist_loss
                    
                    
                    gpu_src_losses += [gpu_src_loss]

                    gpu_G_loss_gvs += [ nn.gradients ( gpu_src_loss, self.model_trainable_weights ) ]
                    
            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                src_rec  = nn.concat(gpu_src_rec_list, 0)
                gauss_mu = nn.concat(gauss_mu_list, 0)
                src_loss = tf.concat(gpu_src_losses, 0)
                loss_gv_op = self.model_opt.get_update_op (nn.average_gv_list (gpu_G_loss_gvs))

            # Initializing training and view functions
            def ae_train(warped_src, target_src):
                s, _ = nn.tf_sess.run ( [ src_loss, loss_gv_op], feed_dict={self.warped_src:warped_src, self.target_src:target_src})
                return s
            self.ae_train = ae_train

            def AE_view(warped_src, target_src):
                return nn.tf_sess.run ( [src_rec, gauss_mu], feed_dict={self.warped_src:warped_src, self.target_src:target_src})
            self.AE_view = AE_view
            
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                if 'df' in archi:
                    gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                    _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            def AE_merge( warped_dst):
                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

            self.AE_merge = AE_merge

        # Loading/initializing all models/optimizers weights
        for model, filename in io.progress_bar_generator(self.model_filename_list, "Initializing models"):
            do_init = self.is_first_run()
            if not do_init:
                do_init = not model.load_weights( self.get_strpath_storage_for_file(filename) )
            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path
            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size()*2,
                        sample_process_options=SampleProcessor.Options(random_flip=False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR, 'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                              ],
                        generators_count=src_generators_count ),
                             ])

    #override
    def get_model_filename_list(self):
        return self.model_filename_list

    #override
    def onSave(self):
        for model, filename in io.progress_bar_generator(self.get_model_filename_list(), "Saving", leave=False):
            model.save_weights ( self.get_strpath_storage_for_file(filename) )


    #override
    def onTrainOneIter(self):
        bs = self.get_batch_size()

        ( (target_src, check_src), ) = self.generate_next_samples()
        
        warped_src = target_src[0:bs]
        target_src = target_src[bs:]
        
        src_loss = self.ae_train(warped_src, target_src)
        src_loss += self.ae_train(target_src, warped_src)
        src_loss /= 2

        return ( ('src_loss', np.mean(src_loss) ), )

    #override
    def onGetPreview(self, samples):
        bs = self.get_batch_size()
        ( (target_src, check_src), ) = samples
        
        warped_src = check_src[0:bs]
        target_src = check_src[bs:]

        S, SS, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([warped_src,target_src]) ]
        RS, GS = self.AE_view (warped_src, target_src)
        RS, = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([RS]) ]
        
        #import code
        #code.interact(local=dict(globals(), **locals()))

        st = []
        for i in range(min(4, bs)):
            
            #map = np.zeros( (self.resolution,self.resolution,3), dtype=np.float32 )
            
            rs_map = RS[i].copy()
            for n in range(self.landmarks_count):
                x,y = GS[i, n]
                color = colorsys.hsv_to_rgb ( n * (1.0/self.landmarks_count), 1.0, 1.0 )
                #cv2.circle (map, ( int( ((x+1)/2.0)*self.resolution), int( ((y+1)/2.0)*self.resolution) ), 1, color, lineType=cv2.LINE_AA )
                cv2.circle (rs_map, ( int( ((x+1)/2.0)*self.resolution), int( ((y+1)/2.0)*self.resolution) ), 1, color, lineType=cv2.LINE_AA )

            ar = S[i], SS[i], RS[i], rs_map
            
            st.append ( np.concatenate ( ar, axis=1) )

        return [ ('TEST', np.concatenate (st, axis=0 )), ]

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]

        return bgr[0], mask_src_dstm[0][...,0], mask_dst_dstm[0][...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.options['resolution'], self.options['resolution'], 3), merger.MergerConfigMasked(face_type=self.face_type, default_mode = 'overlay')

Model = SAEHDModel

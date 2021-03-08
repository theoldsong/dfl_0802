import multiprocessing
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
from facelib import FaceType
from models import ModelBase
from samplelib import *

class QModel(ModelBase):
    #override
    def on_initialize(self):
        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices
        self.model_data_format = "NCHW" if len(devices) != 0 and not self.is_debug() else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        resolution = self.resolution = 96
        self.face_type = FaceType.FULL
        ae_dims = 256
        e_dims = 64
        d_dims = 64
        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch = 3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        self.model_filename_list = []

        kernel_initializer=tf.initializers.glorot_uniform()
        class Upscale(nn.ModelBase):
            def on_build(self, in_ch, out_ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME',kernel_initializer=kernel_initializer)

            def forward(self, x):
                x = self.conv1(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = nn.depth_to_space(x, 2)
                return x

        class ResidualBlock(nn.ModelBase):
            def on_build(self, ch, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME',kernel_initializer=kernel_initializer)
                self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME',kernel_initializer=kernel_initializer)

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = tf.nn.leaky_relu(inp + x, 0.2)
                return x

        class Encoder(nn.ModelBase):
            def on_build(self, in_ch, e_ch):
                
                
                self.down11 = nn.Conv2D( in_ch, e_ch, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down12 = nn.Conv2D( e_ch, e_ch, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                
                self.down21 = nn.Conv2D( e_ch, e_ch*2, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down22 = nn.Conv2D( e_ch*2, e_ch*2, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)                
                
                self.down31 = nn.Conv2D( e_ch*2, e_ch*4, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down32 = nn.Conv2D( e_ch*4, e_ch*4, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down33 = nn.Conv2D( e_ch*4, e_ch*4, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                
                self.down41 = nn.Conv2D( e_ch*4, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down42 = nn.Conv2D( e_ch*8, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down43 = nn.Conv2D( e_ch*8, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                
                self.down51 = nn.Conv2D( e_ch*8, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down52 = nn.Conv2D( e_ch*8, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)
                self.down53 = nn.Conv2D( e_ch*8, e_ch*8, kernel_size=3, strides=1, padding='SAME',kernel_initializer=kernel_initializer)

            def forward(self, inp):
                x = inp
                
                x = self.down11(x)
                x = self.down12(x)
                x = nn.max_pool(x)
                
                x = self.down21(x)
                x = self.down22(x)
                x = nn.max_pool(x)
                
                x = self.down31(x)
                x = self.down32(x)
                x = self.down33(x)
                x = nn.max_pool(x)
                
                x = self.down41(x)
                x = self.down42(x)
                x = self.down43(x)
                x = nn.max_pool(x)
                
                x = self.down51(x)
                x = self.down52(x)
                x = self.down53(x)
                x = nn.max_pool(x)
                
                x = nn.flatten(x)
                return x
        
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
                                        padding='SAME', dilations=self.dilations,kernel_initializer=kernel_initializer)

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
                self.conv1 = nn.Conv2D( in_ch, out_ch*4, kernel_size=kernel_size, padding='SAME',kernel_initializer=kernel_initializer)

            def forward(self, x):
                x = self.conv1(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = nn.depth_to_space(x, 2)
                return x

        class Encoder(nn.ModelBase):
            def on_build(self, in_ch, e_ch):
                self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales=4, kernel_size=5, dilations=1, subpixel=False)

            def forward(self, inp):
                x = nn.flatten(self.down1(inp))
                return x
                    
        class Branch(nn.ModelBase):
            def on_build(self, in_ch, ae_ch):
                self.dense1 = nn.Dense( in_ch, ae_ch )

            def forward(self, inp):
                x = self.dense1(inp)
                return x

        class Classifier(nn.ModelBase):
            def on_build(self, in_ch, n_classes):
                self.dense1 = nn.Dense( in_ch, 4096 )
                self.dense2 = nn.Dense( 4096, 4096 )
                self.pitch_dense = nn.Dense( 4096, n_classes )
                self.yaw_dense = nn.Dense( 4096, n_classes )


            def forward(self, inp):
                x = inp
                x = self.dense1(x)
                x = self.dense2(x)
                return self.pitch_dense(x), self.yaw_dense(x)


        lowest_dense_res = resolution // 16

        class Inter(nn.ModelBase):
            def on_build(self, in_ch, ae_out_ch):
                self.ae_out_ch = ae_out_ch

                self.dense2 = nn.Dense( in_ch, lowest_dense_res * lowest_dense_res * ae_out_ch )
                self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

            def forward(self, inp):
                x = inp
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

                self.res0 = ResidualBlock(d_ch*8, kernel_size=3)
                self.res1 = ResidualBlock(d_ch*4, kernel_size=3)
                self.res2 = ResidualBlock(d_ch*2, kernel_size=3)

                self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME',kernel_initializer=kernel_initializer)

            def forward(self, inp):
                z = inp

                x = self.upscale0(z)
                x = self.res0(x)
                x = self.upscale1(x)
                x = self.res1(x)
                x = self.upscale2(x)
                x = self.res2(x)

                return tf.nn.sigmoid(self.out_conv(x))

        n_pyr_degs = self.n_pyr_degs = 3
        n_pyr_classes = self.n_pyr_classes = 180 // self.n_pyr_degs
        
        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)
            self.pitches_vector = tf.placeholder (nn.floatx, (None,n_pyr_classes) )
            self.yaws_vector = tf.placeholder (nn.floatx, (None,n_pyr_classes) )
            

        
        
        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')
            encoder_out_ch = self.encoder.compute_output_channels ( (nn.floatx, bgr_shape))

            self.bT = Branch (in_ch=encoder_out_ch, ae_ch=ae_dims,  name='bT')
            self.bP = Branch (in_ch=encoder_out_ch, ae_ch=ae_dims,  name='bP')

            self.bTC = Classifier(in_ch=ae_dims, n_classes=self.n_pyr_classes, name='bTC')
            self.bPC = Classifier(in_ch=ae_dims, n_classes=self.n_pyr_classes, name='bPC')

            self.inter = Inter (in_ch=ae_dims*2, ae_out_ch=ae_dims*2, name='inter')

            self.decoder = Decoder(in_ch=ae_dims*2, d_ch=d_dims, d_mask_ch=d_dims, name='decoder')


            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.bT,       'bT.npy'      ],
                                          [self.bTC,       'bTC.npy'      ],
                                          [self.bP,       'bP.npy'      ],
                                          [self.bPC,       'bPC.npy'      ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder, 'decoder.npy']  ]

            if self.is_training:
                self.all_trainable_weights = self.encoder.get_weights() + \
                                             self.bT.get_weights() +\
                                             self.bTC.get_weights() +\
                                             self.bP.get_weights() +\
                                             self.bPC.get_weights() +\
                                             self.inter.get_weights() +\
                                             self.decoder.get_weights()

                # Initialize optimizers
                self.src_dst_opt = nn.RMSprop(lr=5e-5, name='src_dst_opt')
                self.src_dst_opt.initialize_variables(self.all_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu )
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, 32 // gpu_count)
            self.set_batch_size( gpu_count*bs_per_gpu)

            # Compute losses per GPU
            gpu_pred_src_list = []
            gpu_pred_dst_list = []

            gpu_A_losses = []
            gpu_B_losses = []
            gpu_C_losses = []
            gpu_D_losses = []
            gpu_A_loss_gvs = []
            gpu_B_loss_gvs = []
            gpu_C_loss_gvs = []
            gpu_D_loss_gvs = []
            for gpu_id in range(gpu_count):
                with tf.device( f'/GPU:{gpu_id}' if len(devices) != 0 else f'/CPU:0' ):
                    batch_slice = slice( gpu_id*bs_per_gpu, (gpu_id+1)*bs_per_gpu )
                    with tf.device(f'/CPU:0'):
                        # slice on CPU, otherwise all batch data will be transfered to GPU first
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        
                        gpu_pitches_vector = self.pitches_vector [batch_slice,:]
                        gpu_yaws_vector    = self.yaws_vector[batch_slice,:]
                        
                    # process model tensors
                    gpu_src_enc_code    = self.encoder(gpu_warped_src)
                    gpu_dst_enc_code    = self.encoder(gpu_target_dst)
                    
                    gpu_src_bT_code     = self.bT(gpu_src_enc_code)
                    gpu_src_bT_code_ng = tf.stop_gradient(gpu_src_bT_code)
                    
                    gpu_src_T_pitch, gpu_src_T_yaw = self.bTC(gpu_src_bT_code)
                    
                    gpu_dst_bT_code     = self.bT(gpu_dst_enc_code)

                    gpu_src_bP_code     = self.bP(gpu_src_enc_code)
                    
                    gpu_src_P_pitch, gpu_src_P_yaw       = self.bPC(gpu_src_bP_code)

                    def crossentropy(target, output):
                        output = tf.nn.softmax(output)
                        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
                        return tf.reduce_sum(target * -tf.log(output), axis=-1, keepdims=False)

                    def negative_crossentropy(n_classes, output):
                        output = tf.nn.softmax(output)
                        output = tf.clip_by_value(output, 1e-7, 1 - 1e-7)
                        return (1.0/n_classes) * tf.reduce_sum(tf.log(output), axis=-1, keepdims=False)


                    gpu_src_bT_code_n = gpu_src_bT_code_ng + tf.random.normal( tf.shape(gpu_src_bT_code_ng) )
                    gpu_src_bP_code_n = gpu_src_bP_code    + tf.random.normal( tf.shape(gpu_src_bP_code) )

                    gpu_pred_src   = self.decoder (self.inter(tf.concat([gpu_src_bT_code_ng, gpu_src_bP_code], axis=-1)))
                    gpu_pred_src_n = self.decoder (self.inter(tf.concat([gpu_src_bT_code_n, gpu_src_bP_code_n], axis=-1)))
                    gpu_pred_dst = self.decoder (self.inter(tf.concat([gpu_dst_bT_code, gpu_src_bP_code], axis=-1)))

                    gpu_A_loss  = 1.0*crossentropy(gpu_pitches_vector, gpu_src_T_pitch ) + \
                                  1.0*crossentropy(gpu_yaws_vector,    gpu_src_T_yaw )
                    
                    
                    gpu_B_loss = 0.1*crossentropy(gpu_pitches_vector, gpu_src_P_pitch ) + \
                                 0.1*crossentropy(gpu_yaws_vector, gpu_src_P_yaw )
                    
                    gpu_C_loss = 0.1*negative_crossentropy( n_pyr_classes, gpu_src_P_pitch ) + \
                                 0.1*negative_crossentropy( n_pyr_classes, gpu_src_P_yaw )
                        

                    gpu_D_loss = 0.0000001*(\
                                    0.5*tf.reduce_sum(tf.square(gpu_target_src-gpu_pred_src), axis=[1,2,3]) + \
                                    0.5*tf.reduce_sum(tf.square(gpu_target_src-gpu_pred_src_n), axis=[1,2,3]) )

                    gpu_pred_src_list.append(gpu_pred_src)
                    gpu_pred_dst_list.append(gpu_pred_dst)
            
                    gpu_A_losses += [gpu_A_loss]
                    gpu_B_losses += [gpu_B_loss]
                    gpu_C_losses += [gpu_C_loss]                    
                    gpu_D_losses += [gpu_D_loss]
                    
                    A_weights = self.encoder.get_weights() + self.bT.get_weights() + self.bTC.get_weights()
                    B_weights = self.bPC.get_weights()
                    C_weights = self.encoder.get_weights() + self.bP.get_weights()
                    D_weights = self.inter.get_weights() + self.decoder.get_weights()
                    
                    
                    gpu_A_loss_gvs += [ nn.gradients ( gpu_A_loss, A_weights ) ]
                    gpu_B_loss_gvs += [ nn.gradients ( gpu_B_loss, B_weights ) ]
                    gpu_C_loss_gvs += [ nn.gradients ( gpu_C_loss, C_weights ) ]
                    gpu_D_loss_gvs += [ nn.gradients ( gpu_D_loss, D_weights ) ]

            # Average losses and gradients, and create optimizer update ops
            with tf.device (models_opt_device):
                pred_src  = nn.concat(gpu_pred_src_list, 0)
                pred_dst  = nn.concat(gpu_pred_dst_list, 0)
                A_loss = nn.average_tensor_list(gpu_A_losses)
                B_loss = nn.average_tensor_list(gpu_B_losses)
                C_loss = nn.average_tensor_list(gpu_C_losses)
                D_loss = nn.average_tensor_list(gpu_D_losses)

                A_loss_gv = nn.average_gv_list (gpu_A_loss_gvs)
                B_loss_gv = nn.average_gv_list (gpu_B_loss_gvs)
                C_loss_gv = nn.average_gv_list (gpu_C_loss_gvs)
                D_loss_gv = nn.average_gv_list (gpu_D_loss_gvs)
                A_loss_gv_op = self.src_dst_opt.get_update_op (A_loss_gv)
                B_loss_gv_op = self.src_dst_opt.get_update_op (B_loss_gv)
                C_loss_gv_op = self.src_dst_opt.get_update_op (C_loss_gv)
                D_loss_gv_op = self.src_dst_opt.get_update_op (D_loss_gv)

            # Initializing training and view functions
            def A_train(warped_src, target_src, pitches_vector, yaws_vector):
                l, _ = nn.tf_sess.run ( [ A_loss, A_loss_gv_op], feed_dict={self.warped_src :warped_src, self.target_src :target_src, self.pitches_vector:pitches_vector, self.yaws_vector:yaws_vector})
                return np.mean(l)
            self.A_train = A_train
            
            def B_train(warped_src, target_src, pitches_vector, yaws_vector):
                l, _ = nn.tf_sess.run ( [ B_loss, B_loss_gv_op], feed_dict={self.warped_src :warped_src, self.target_src :target_src, self.pitches_vector:pitches_vector, self.yaws_vector:yaws_vector})
                return np.mean(l)
            self.B_train = B_train
            
            def C_train(warped_src, target_src, pitches_vector, yaws_vector):
                l, _ = nn.tf_sess.run ( [ C_loss, C_loss_gv_op], feed_dict={self.warped_src :warped_src, self.target_src :target_src, self.pitches_vector:pitches_vector, self.yaws_vector:yaws_vector})
                return np.mean(l)
            self.C_train = C_train
            
            def D_train(warped_src, target_src, pitches_vector, yaws_vector):
                l, _ = nn.tf_sess.run ( [ D_loss, D_loss_gv_op], feed_dict={self.warped_src :warped_src, self.target_src :target_src, self.pitches_vector:pitches_vector, self.yaws_vector:yaws_vector})
                return np.mean(l)
            self.D_train = D_train

            def AE_view(warped_src):
                return nn.tf_sess.run ([pred_src], feed_dict={self.warped_src:warped_src})

            self.AE_view = AE_view
            
            def AE_view2(warped_src, target_dst):
                return nn.tf_sess.run ([pred_dst], feed_dict={self.warped_src:warped_src, self.target_dst:target_dst})

            self.AE_view2 = AE_view2
        else:
            # Initializing merge function
            with tf.device( f'/GPU:0' if len(devices) != 0 else f'/CPU:0'):
                gpu_dst_code     = self.inter(self.encoder(self.warped_dst))
                gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code)
                _, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code)

            def AE_merge( warped_dst):

                return nn.tf_sess.run ( [gpu_pred_src_dst, gpu_pred_dst_dstm, gpu_pred_src_dstm], feed_dict={self.warped_dst:warped_dst})

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

            if do_init and self.pretrained_model_path is not None:
                pretrained_filepath = self.pretrained_model_path / filename
                if pretrained_filepath.exists():
                    do_init = not model.load_weights(pretrained_filepath)

            if do_init:
                model.init_weights()

        # initializing sample generators
        if self.is_training:
            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.PITCH_YAW_ROLL_SIGMOID, 'resolution': resolution},                                                
                                                ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':True,  'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.FACE_IMAGE,'warp':False, 'transform':True, 'channel_type' : SampleProcessor.ChannelType.BGR,                                                           'face_type':self.face_type, 'data_format':nn.data_format, 'resolution': resolution},
                                                {'sample_type': SampleProcessor.SampleType.PITCH_YAW_ROLL_SIGMOID, 'resolution': resolution},     
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
        bs = self.get_batch_size()
        
        samples = self.last_samples = self.generate_next_samples()
        ( (warped_src, target_src, pyr_src), \
          (warped_dst, target_dst, pyr_dst) ) = samples
        
        pitches = ( pyr_src[:,0] / (1/self.n_pyr_classes) ).astype(np.int32)
        pitches_vector = np.zeros( (len(pitches), self.n_pyr_classes), np.float32)
        pitches_vector[np.arange(bs), pitches] = 1.0

        yaws = ( pyr_src[:,1] / (1/self.n_pyr_classes) ).astype(np.int32)
        yaws_vector = np.zeros( (len(yaws), self.n_pyr_classes), np.float32)
        yaws_vector[np.arange(bs), yaws] = 1.0
          
        #import code
        #code.interact(local=dict(globals(), **locals()))


        A_loss = self.A_train (warped_src, target_src, pitches_vector, yaws_vector)

        B_loss = self.B_train (warped_src, target_src, pitches_vector, yaws_vector)

        C_loss = self.C_train (warped_src, target_src, pitches_vector, yaws_vector)

        D_loss = self.D_train (warped_src, target_src, pitches_vector, yaws_vector)
                      
        return ( ('A_loss', A_loss), ('B_loss', B_loss), ('C_loss', C_loss), ('D_loss', D_loss), )

    #override
    def onGetPreview(self, samples):
        ( (warped_src, target_src, pyr_src),
          (warped_dst, target_dst, pyr_dst) ) = samples

        S, D, SS, DD, DS = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + \
                                                  self.AE_view (target_src) + \
                                                  self.AE_view (target_dst)  + \
                                                  self.AE_view2 (target_dst, target_src) 
                                                  ) ]

        n_samples = min(4, self.get_batch_size() )
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], DS[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('TEST', np.concatenate (st, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]
        mask = mask_dst_dstm[0] * mask_src_dstm[0]
        return bgr[0], mask[...,0]

    #override
    def get_MergerConfig(self):
        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=self.face_type,
                                     default_mode = 'overlay',
                                    )

Model = QModel

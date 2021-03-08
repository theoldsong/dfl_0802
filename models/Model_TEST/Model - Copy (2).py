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
        self.model_data_format = "NCHW" if len(device_config.devices) != 0 and not self.is_debug()  else "NHWC"
        nn.initialize(data_format=self.model_data_format)
        tf = nn.tf

        conv_kernel_initializer = nn.initializers.ca()

        class Downscale(nn.ModelBase):
            def __init__(self, in_ch, out_ch, kernel_size=3, dilations=1, use_activator=True, *kwargs ):
                self.in_ch = in_ch
                self.out_ch = out_ch
                self.kernel_size = kernel_size
                self.dilations = dilations
                self.use_activator = use_activator
                super().__init__(*kwargs)

            def on_build(self, *args, **kwargs ):
                self.conv1 = nn.Conv2D( self.in_ch,
                                          self.out_ch,
                                          kernel_size=self.kernel_size,
                                          strides=2,
                                          padding='SAME', dilations=self.dilations, kernel_initializer=conv_kernel_initializer )

            def forward(self, x):
                x = self.conv1(x)
                if self.use_activator:
                    x = tf.nn.leaky_relu(x, 0.1)
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
            def on_build(self, ch, mod=1, kernel_size=3 ):
                self.conv1 = nn.Conv2D( ch, ch*mod, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)
                self.conv2 = nn.Conv2D( ch*mod, ch, kernel_size=kernel_size, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, inp):
                x = self.conv1(inp)
                x = tf.nn.leaky_relu(x, 0.1)
                x = self.conv2(x)
                x = inp + x
                x = tf.nn.leaky_relu(x, 0.1)
                return x

        class Encoder(nn.ModelBase):
            def on_build(self, in_ch, e_ch):
                self.conv1 = Downscale(in_ch, e_ch)
                self.conv2 = Downscale(e_ch, e_ch*2 )
                self.conv3 = Downscale(e_ch*2, e_ch*4)
                self.conv4 = Downscale(e_ch*4, e_ch*8)
                self.conv5 = Downscale(e_ch*8, e_ch*16)
                self.conv6 = Downscale(e_ch*16, e_ch*32)
                self.conv7 = Downscale(e_ch*32, e_ch*64)
                
                self.res1 = ResidualBlock(e_ch)
                self.res2 = ResidualBlock(e_ch*2)
                self.res3 = ResidualBlock(e_ch*4)
                self.res4 = ResidualBlock(e_ch*8)
                self.res5 = ResidualBlock(e_ch*16)
                self.res6 = ResidualBlock(e_ch*32)
                self.res7 = ResidualBlock(e_ch*64)
                
            def forward(self, inp):
                x = self.conv1 (inp)
                x = self.res1(x)
                x = self.conv2 (x)
                x = self.res2(x)
                x = self.conv3 (x)
                x = self.res3(x)
                x = self.conv4 (x)
                x = self.res4(x)
                x = self.conv5 (x)
                x = self.res5(x)
                x = self.conv6 (x)
                x = self.res6(x)
                x = self.conv7 (x)                
                x = self.res7(x)
                return x
                

        class Inter(nn.ModelBase):
            def __init__(self, in_ch, ae_ch, **kwargs):
                self.in_ch, self.ae_ch = in_ch, ae_ch
                super().__init__(**kwargs)

            def on_build(self):
                in_ch, ae_ch = self.in_ch, self.ae_ch                
                
                self.dense_conv1 = nn.Conv2D( in_ch, 64, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)
                self.dense_conv2 = nn.Conv2D( 64, in_ch, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)
             
                
                self.conv7 = Upscale(in_ch, in_ch//2)
                self.conv6 = Upscale(in_ch//2, in_ch//4)
                
            def forward(self, inp):
                x = inp
                x = self.dense_conv1(x)
                x = self.dense_conv2(x)
                x = self.conv7(x)
                x = self.conv6(x)
                                
                return x


        class Decoder(nn.ModelBase):
            def on_build(self, in_ch):
                self.upscale6 = Upscale(in_ch, in_ch//2)
                self.upscale5 = Upscale(in_ch//2, in_ch//4)
                self.upscale4 = Upscale(in_ch//4, in_ch//8)
                self.upscale3 = Upscale(in_ch//8, in_ch//16)
                self.upscale2 = Upscale(in_ch//16, in_ch//32)
                #self.upscale1 = Upscale(in_ch//32, in_ch//64)
                self.out_conv  = nn.Conv2D(  in_ch//32, 3, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)
                
                self.res61 = ResidualBlock(in_ch//2,  mod=8)
                self.res62 = ResidualBlock(in_ch//2,  mod=8)
                self.res63 = ResidualBlock(in_ch//2,  mod=8)
                self.res51 = ResidualBlock(in_ch//4,  mod=8)
                self.res52 = ResidualBlock(in_ch//4,  mod=8)
                self.res53 = ResidualBlock(in_ch//4,  mod=8)
                self.res41 = ResidualBlock(in_ch//8,  mod=8)
                self.res42 = ResidualBlock(in_ch//8,  mod=8)
                self.res43 = ResidualBlock(in_ch//8,  mod=8)                
                self.res31 = ResidualBlock(in_ch//16, mod=8)
                self.res32 = ResidualBlock(in_ch//16, mod=8)
                self.res33 = ResidualBlock(in_ch//16, mod=8)                
                self.res21 = ResidualBlock(in_ch//32, mod=8)
                self.res22 = ResidualBlock(in_ch//32, mod=8)
                self.res23 = ResidualBlock(in_ch//32, mod=8)
                
                m_ch = in_ch//2
                self.upscalem6 = Upscale(in_ch, m_ch//2)
                self.upscalem5 = Upscale(m_ch//2, m_ch//4)
                self.upscalem4 = Upscale(m_ch//4, m_ch//8)
                self.upscalem3 = Upscale(m_ch//8, m_ch//16)
                self.upscalem2 = Upscale(m_ch//16, m_ch//32)
                #self.upscalem1 = Upscale(m_ch//32, m_ch//64)                
                self.out_convm = nn.Conv2D(  m_ch//32, 1, kernel_size=1, padding='SAME', kernel_initializer=conv_kernel_initializer)

            def forward(self, inp):
                z = inp
                x = self.upscale6 (z)
                x = self.res61(x)
                x = self.res62(x)
                x = self.res63(x)
                x = self.upscale5 (x)
                x = self.res51(x)
                x = self.res52(x)
                x = self.res53(x)
                x = self.upscale4 (x)
                x = self.res41(x)
                x = self.res42(x)
                x = self.res43(x)
                x = self.upscale3 (x)
                x = self.res31(x)
                x = self.res32(x)
                x = self.res33(x)
                x = self.upscale2 (x)
                x = self.res21(x)
                x = self.res22(x)
                x = self.res23(x)
                #x = self.upscale1 (x)

                y = self.upscalem6 (z)
                y = self.upscalem5 (y)
                y = self.upscalem4 (y)
                y = self.upscalem3 (y)
                y = self.upscalem2 (y)
                #y = self.upscalem1 (y)

                return tf.nn.sigmoid(self.out_conv(x)), \
                       tf.nn.sigmoid(self.out_convm(y))

        device_config = nn.getCurrentDeviceConfig()
        devices = device_config.devices

        resolution = self.resolution = 128
        ae_dims = 128
        e_dims = 16
        
        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        models_opt_on_gpu = len(devices) >= 1 and all([dev.total_mem_gb >= 4 for dev in devices])
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_ch = 3
        output_ch = 3
        bgr_shape = nn.get4Dshape(resolution,resolution,input_ch)
        mask_shape = nn.get4Dshape(resolution,resolution,1)

        self.model_filename_list = []


        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (nn.floatx, bgr_shape)
            self.warped_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_src = tf.placeholder (nn.floatx, bgr_shape)
            self.target_dst = tf.placeholder (nn.floatx, bgr_shape)

            self.target_srcm = tf.placeholder (nn.floatx, mask_shape)
            self.target_dstm = tf.placeholder (nn.floatx, mask_shape)

        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(in_ch=input_ch, e_ch=e_dims, name='encoder')

            self.inter = Inter (in_ch=e_dims*64, ae_ch=ae_dims, name='inter')

            self.decoder_src = Decoder(in_ch=e_dims*16, name='decoder_src')
            self.decoder_dst = Decoder(in_ch=e_dims*16, name='decoder_dst')

            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                self.src_dst_trainable_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()

                # Initialize optimizers
                self.src_dst_opt = nn.RMSprop(lr=5e-5, name='src_dst_opt')
                self.src_dst_opt.initialize_variables(self.src_dst_trainable_weights, vars_on_cpu=optimizer_vars_on_cpu )
                self.model_filename_list += [ (self.src_dst_opt, 'src_dst_opt.npy') ]

        if self.is_training:
            # Adjust batch size for multiple GPU
            gpu_count = max(1, len(devices) )
            bs_per_gpu = max(1, 4 // gpu_count)
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
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst   = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm  = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_dstm  = self.target_dstm[batch_slice,:,:,:]

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

                    gpu_target_srcm_blur = nn.gaussian_blur(gpu_target_srcm,  max(1, resolution // 32) )
                    gpu_target_dstm_blur = nn.gaussian_blur(gpu_target_dstm,  max(1, resolution // 32) )

                    gpu_target_dst_masked      = gpu_target_dst*gpu_target_dstm_blur
                    gpu_target_dst_anti_masked = gpu_target_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_target_src_masked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_src_masked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_src_masked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square( gpu_target_srcm - gpu_pred_src_srcm ),axis=[1,2,3] )

                    gpu_dst_loss  = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_G_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [ nn.gradients ( gpu_G_loss, self.src_dst_trainable_weights ) ]


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
                src_dst_loss_gv = nn.average_gv_list (gpu_src_dst_loss_gvs)
                src_dst_loss_gv_op = self.src_dst_opt.get_update_op (src_dst_loss_gv)

            # Initializing training and view functions
            def src_dst_train(warped_src, target_src, target_srcm, \
                              warped_dst, target_dst, target_dstm):
                s, d, _ = nn.tf_sess.run ( [ src_loss, dst_loss, src_dst_loss_gv_op],
                                            feed_dict={self.warped_src :warped_src,
                                                       self.target_src :target_src,
                                                       self.target_srcm:target_srcm,
                                                       self.warped_dst :warped_dst,
                                                       self.target_dst :target_dst,
                                                       self.target_dstm:target_dstm,
                                                       })
                s = np.mean(s)
                d = np.mean(d)
                return s, d
            self.src_dst_train = src_dst_train

            def AE_view(warped_src, warped_dst):
                return nn.tf_sess.run ( [pred_src_src, pred_dst_dst, pred_dst_dstm, pred_src_dst, pred_src_dstm],
                                            feed_dict={self.warped_src:warped_src,
                                                    self.warped_dst:warped_dst})

            self.AE_view = AE_view
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
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = min(multiprocessing.cpu_count(), 8)
            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count // 2

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'data_format':nn.data_format, 'resolution':resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR),        'data_format':nn.data_format, 'resolution': resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_HULL),          'data_format':nn.data_format, 'resolution': resolution } ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'data_format':nn.data_format, 'resolution':resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR),        'data_format':nn.data_format, 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_HULL),          'data_format':nn.data_format, 'resolution': resolution} ],
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

        samples = self.generate_next_samples()
        ( (warped_src, target_src, target_srcm), \
          (warped_dst, target_dst, target_dstm) ) = samples

        src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm,
                                                 warped_dst, target_dst, target_dstm)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples):
        ( (warped_src, target_src, target_srcm),
          (warped_dst, target_dst, target_dstm) ) = samples

        S, D, SS, DD, DDM, SD, SDM = [ np.clip( nn.to_data_format(x,"NHWC", self.model_data_format), 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

        target_srcm, target_dstm = [ nn.to_data_format(x,"NHWC", self.model_data_format) for x in ([target_srcm, target_dstm] )]

        n_samples = min(4, self.get_batch_size() )
        result = []
        st = []
        for i in range(n_samples):
            ar = S[i], SS[i], D[i], DD[i], SD[i]
            st.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96', np.concatenate (st, axis=0 )), ]

        st_m = []
        for i in range(n_samples):
            ar = S[i]*target_srcm[i], SS[i], D[i]*target_dstm[i], DD[i]*DDM[i], SD[i]*(DDM[i]*SDM[i])
            st_m.append ( np.concatenate ( ar, axis=1) )

        result += [ ('Quick96 masked', np.concatenate (st_m, axis=0 )), ]

        return result

    def predictor_func (self, face=None):
        face = nn.to_data_format(face[None,...], self.model_data_format, "NHWC")

        bgr, mask_dst_dstm, mask_src_dstm = [ nn.to_data_format(x, "NHWC", self.model_data_format).astype(np.float32) for x in self.AE_merge (face) ]
        mask = mask_dst_dstm[0] * mask_src_dstm[0]
        return bgr[0], mask[...,0]

    #override
    def get_MergerConfig(self):
        face_type = FaceType.FULL

        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=face_type,
                                     default_mode = 'overlay',
                                    )

Model = QModel

import multiprocessing
from functools import partial

import numpy as np

from core import mathlib
from core.interact import interact as io
from core.leras import nn
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
        nn.initialize()
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

        class Encoder(nn.ModelBase):
            def on_build(self, e_ch, levels):
                self.enc_blocks = {}
                self.from_rgbs = {}
                self.dense_norm = nn.DenseNorm()

                in_ch = e_ch
                out_ch = in_ch
                for level in range(levels,-1,-1):
                    self.max_ch = out_ch = np.clip(out_ch*2, 0, 512)

                    self.enc_blocks[level] = EncBlock(in_ch, out_ch, level)
                    self.from_rgbs[level] = FromRGB(in_ch)

                    in_ch = out_ch


            def forward(self, inp, stage):
                x = inp

                for level in range(stage, -1, -1):
                    if stage in self.enc_blocks:
                        if level == stage:
                            x = self.from_rgbs[level](x)
                        x = self.enc_blocks[level](x)

                x = nn.flatten(x)
                x = self.dense_norm(x)
                x = nn.reshape_4D (x, 1, 1, self.max_ch)
                
                return x

            def get_stage_weights(self, stage):
                self.get_weights()
                weights = []
                for level in range(stage, -1, -1):
                    if stage in self.enc_blocks:
                        if level == stage:
                            weights.append ( self.from_rgbs[level].get_weights() )
                        weights.append ( self.enc_blocks[level].get_weights() )

                if len(weights) == 0:
                    return []
                elif len(weights) == 1:
                    return weights[0]
                else:
                    return sum(weights[1:],weights[0])
                


        class Decoder(nn.ModelBase):
            def on_build(self, d_ch, total_levels, levels_range):
                
                self.dec_blocks = {}
                self.to_rgbs = {}
                
                level_ch = {}
                ch = d_ch
                for level in range(total_levels,-2,-1):
                    level_ch[level] = ch
                    ch = np.clip(ch*2, 0, 512)
                
                out_ch = level_ch[levels_range[1]]
                for level in range(levels_range[1],levels_range[0]-1,-1):
                    in_ch = level_ch[level-1]
                    
                    self.dec_blocks[level] = DecBlock(in_ch, out_ch, level)
                    self.to_rgbs[level] = ToRGB(out_ch)

                    out_ch = in_ch
                    
            def forward(self, inp, stage):
                x = inp
                
                for level in range(stage+1):
                    if level in self.dec_blocks:                    
                        x = self.dec_blocks[level](x)
                        if level == stage:
                            x = self.to_rgbs[level](x)
                return x
                
            def get_stage_weights(self, stage):
                # Call internal get_weights in order to initialize inner logic
                self.get_weights()

                weights = []
                for level in range(stage+1):
                    if level in self.dec_blocks:
                        weights.append ( self.dec_blocks[level].get_weights() )
                        if level == stage:
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

        stage_resolutions = [ 2**(i+2) for i in range(self.stage_max+1) ]        
        stage_resolution = stage_resolutions[stage]
        
        ed_dims = 16
        
        self.pretrain = False
        self.pretrain_just_disabled = False

        masked_training = True

        models_opt_on_gpu = len(devices) == 1 and devices[0].total_mem_gb >= 4
        models_opt_device = '/GPU:0' if models_opt_on_gpu and self.is_training else '/CPU:0'
        optimizer_vars_on_cpu = models_opt_device=='/CPU:0'

        input_nc = 3
        output_nc = 3
        bgr_shape = (stage_resolution, stage_resolution, output_nc)
        mask_shape = (stage_resolution, stage_resolution, 1)

        self.model_filename_list = []

        with tf.device ('/CPU:0'):
            #Place holders on CPU
            self.warped_src = tf.placeholder (tf.float32, (None,)+bgr_shape)
            self.warped_dst = tf.placeholder (tf.float32, (None,)+bgr_shape)

            self.target_src = tf.placeholder (tf.float32, (None,)+bgr_shape)
            self.target_dst = tf.placeholder (tf.float32, (None,)+bgr_shape)

            self.target_srcm = tf.placeholder (tf.float32, (None,)+mask_shape)
            self.target_dstm = tf.placeholder (tf.float32, (None,)+mask_shape)

        # Initializing model classes
        with tf.device (models_opt_device):
            self.encoder = Encoder(e_ch=ed_dims, levels=self.stage_max, name='encoder')
            
            

            self.inter = Decoder(d_ch=ed_dims, total_levels=self.stage_max, levels_range=[0,2], name='inter')
            self.decoder_src = Decoder(d_ch=ed_dims, total_levels=self.stage_max, levels_range=[3,self.stage_max], name='decoder_src')
            self.decoder_dst = Decoder(d_ch=ed_dims, total_levels=self.stage_max, levels_range=[3,self.stage_max], name='decoder_dst')

            
            
            self.model_filename_list += [ [self.encoder,     'encoder.npy'    ],
                                          [self.inter,       'inter.npy'      ],
                                          [self.decoder_src, 'decoder_src.npy'],
                                          [self.decoder_dst, 'decoder_dst.npy']  ]

            if self.is_training:
                self.src_dst_all_weights = self.encoder.get_weights() + self.inter.get_weights() + self.decoder_src.get_weights() + self.decoder_dst.get_weights()
                self.src_dst_trainable_weights = self.encoder.get_stage_weights(stage) + self.inter.get_stage_weights(stage) \
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
                        gpu_warped_src   = self.warped_src [batch_slice,:,:,:]
                        gpu_warped_dst   = self.warped_dst [batch_slice,:,:,:]
                        gpu_target_src   = self.target_src [batch_slice,:,:,:]
                        gpu_target_dst   = self.target_dst [batch_slice,:,:,:]
                        gpu_target_srcm  = self.target_srcm[batch_slice,:,:,:]
                        gpu_target_dstm  = self.target_dstm[batch_slice,:,:,:]

                    # process model tensors
                    
                    
                    gpu_src_code     = self.inter( self.encoder(gpu_warped_src, stage), stage )
                    gpu_dst_code     = self.inter( self.encoder(gpu_warped_dst, stage), stage )
                    

                    gpu_pred_src_src, gpu_pred_src_srcm = self.decoder_src(gpu_src_code, stage)
                    gpu_pred_dst_dst, gpu_pred_dst_dstm = self.decoder_dst(gpu_dst_code, stage)
                    gpu_pred_src_dst, gpu_pred_src_dstm = self.decoder_src(gpu_dst_code, stage)

                    import code
                    code.interact(local=dict(globals(), **locals()))
                    
                    
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

                    gpu_target_srcmasked_opt  = gpu_target_src*gpu_target_srcm_blur if masked_training else gpu_target_src
                    gpu_target_dst_masked_opt = gpu_target_dst_masked if masked_training else gpu_target_dst

                    gpu_pred_src_src_masked_opt = gpu_pred_src_src*gpu_target_srcm_blur if masked_training else gpu_pred_src_src
                    gpu_pred_dst_dst_masked_opt = gpu_pred_dst_dst*gpu_target_dstm_blur if masked_training else gpu_pred_dst_dst

                    gpu_psd_target_dst_masked = gpu_pred_src_dst*gpu_target_dstm_blur
                    gpu_psd_target_dst_anti_masked = gpu_pred_src_dst*(1.0 - gpu_target_dstm_blur)

                    gpu_src_loss =  tf.reduce_mean ( 10*nn.dssim(gpu_target_srcmasked_opt, gpu_pred_src_src_masked_opt, max_val=1.0, filter_size=int(resolution/11.6)), axis=[1])
                    gpu_src_loss += tf.reduce_mean ( 10*tf.square ( gpu_target_srcmasked_opt - gpu_pred_src_src_masked_opt ), axis=[1,2,3])
                    gpu_src_loss += tf.reduce_mean ( tf.square( gpu_target_srcm - gpu_pred_src_srcm ), axis=[1,2,3] )

                    gpu_dst_loss  = tf.reduce_mean ( 10*nn.dssim(gpu_target_dst_masked_opt, gpu_pred_dst_dst_masked_opt, max_val=1.0, filter_size=int(resolution/11.6) ), axis=[1])
                    gpu_dst_loss += tf.reduce_mean ( 10*tf.square(  gpu_target_dst_masked_opt- gpu_pred_dst_dst_masked_opt ), axis=[1,2,3])
                    gpu_dst_loss += tf.reduce_mean ( tf.square( gpu_target_dstm - gpu_pred_dst_dstm ),axis=[1,2,3] )

                    gpu_src_losses += [gpu_src_loss]
                    gpu_dst_losses += [gpu_dst_loss]

                    gpu_src_dst_loss = gpu_src_loss + gpu_dst_loss
                    gpu_src_dst_loss_gvs += [ nn.gradients ( gpu_src_dst_loss, self.src_dst_trainable_weights ) ]


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
            t = SampleProcessor.Types
            face_type = t.FACE_TYPE_FULL

            training_data_src_path = self.training_data_src_path if not self.pretrain else self.get_pretraining_data_path()
            training_data_dst_path = self.training_data_dst_path if not self.pretrain else self.get_pretraining_data_path()

            cpu_count = multiprocessing.cpu_count()

            src_generators_count = cpu_count // 2
            dst_generators_count = cpu_count - src_generators_count

            self.set_training_data_generators ([
                    SampleGeneratorFace(training_data_src_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution': resolution, },
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_HULL), 'resolution': resolution } ],
                        generators_count=src_generators_count ),

                    SampleGeneratorFace(training_data_dst_path, debug=self.is_debug(), batch_size=self.get_batch_size(),
                        sample_process_options=SampleProcessor.Options(random_flip=True if self.pretrain else False),
                        output_sample_types = [ {'types' : (t.IMG_WARPED_TRANSFORMED, face_type, t.MODE_BGR), 'resolution':resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_BGR), 'resolution': resolution},
                                                {'types' : (t.IMG_TRANSFORMED, face_type, t.MODE_FACE_MASK_ALL_HULL), 'resolution': resolution} ],
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
        if self.get_iter() % 3 == 0 and self.last_samples is not None:
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = self.last_samples
            src_loss, dst_loss = self.src_dst_train (target_src, target_src, target_srcm,
                                                     target_dst, target_dst, target_dstm)
        else:
            samples = self.last_samples = self.generate_next_samples()
            ( (warped_src, target_src, target_srcm), \
              (warped_dst, target_dst, target_dstm) ) = samples

            src_loss, dst_loss = self.src_dst_train (warped_src, target_src, target_srcm,
                                                     warped_dst, target_dst, target_dstm)

        return ( ('src_loss', src_loss), ('dst_loss', dst_loss), )

    #override
    def onGetPreview(self, samples):
        n_samples = min(4, self.get_batch_size() )

        ( (warped_src, target_src, target_srcm),
          (warped_dst, target_dst, target_dstm) ) = \
                [ [sample[0:n_samples] for sample in sample_list ]
                                                 for sample_list in samples ]

        S, D, SS, DD, DDM, SD, SDM = [ np.clip(x, 0.0, 1.0) for x in ([target_src,target_dst] + self.AE_view (target_src, target_dst) ) ]
        DDM, SDM, = [ np.repeat (x, (3,), -1) for x in [DDM, SDM] ]

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

        bgr, mask_dst_dstm, mask_src_dstm = self.AE_merge (face[np.newaxis,...])
        mask = mask_dst_dstm[0] * mask_src_dstm[0]
        return bgr[0], mask[...,0]

    #override
    def get_MergerConfig(self):
        face_type = FaceType.FULL

        import merger
        return self.predictor_func, (self.resolution, self.resolution, 3), merger.MergerConfigMasked(face_type=face_type,
                                     default_mode = 'overlay',
                                    )

Model = TESTModel

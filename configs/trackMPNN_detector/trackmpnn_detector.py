_base_ = './primitive_config.py'

model = dict(
    type='FasterRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=False,
        plugins=[
            dict(
                cfg=dict(type='ContextBlock', ratio=1. / 16),
                stages=(False, True, True, True),
                position='after_conv3')
        ],
        style='pytorch'
    ),

    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],

    bbox_head=dict(
        loss_bbox=dict(
            _delete_=True,
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=0.11,
            loss_weight=1.0)
    )
)


##################### TRAIN CONFIG ######################

train_cfg=dict(
    rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='CombinedSampler',
                num=512,
                pos_fraction=0.25,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(
                    type='IoUBalancedNegSampler',
                    floor_thr=-1,
                    floor_fraction=0,
                    num_bins=3)
            )
        )
)


################### TRAINING PIPELINE ########################

train_pipeline = [  # Training pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(
        type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True,  # Whether to use bounding box, True for detection
        with_mask=False,  # Whether to use instance mask, True for instance segmentation
        poly2mask=False),  # Whether to convert the polygon mask to instance mask, set False for acceleration and to save memory
  
     dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1280, 720),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=True,  # Whether to flip images during testing
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                 keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be supressed by the img_scale set above.
            dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used because flip=False
            dict(
                type='ImageToTensor',  # convert image to tensor
                keys=['img']),
            dict(
                type='Collect',  # Collect pipeline that collect necessary keys for testing.
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
     dict(
        type='MultiScaleFlipAug',  # An encapsulation that encapsulates the testing augmentations
        img_scale=(1280, 720),  # Decides the largest scale for testing, used for the Resize pipeline
        flip=False,  # Whether to flip images during testing
        transforms=[
            dict(type='Resize',  # Use resize augmentation
                 keep_ratio=True),  # Whether to keep the ratio between height and width, the img_scale set here will be supressed by the img_scale set above.
            dict(type='RandomFlip'),  # Thought RandomFlip is added in pipeline, it is not used because flip=False
            dict(
                type='ImageToTensor',  # convert image to tensor
                keys=['img']),
            dict(
                type='Collect',  # Collect pipeline that collect necessary keys for testing.
                keys=['img'])
        ]),
]

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(  # Train dataset config
        type='CocoDataset',  # Type of dataset, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19 for details.
        ann_file='data/coco/annotations/instances_train2017.json',  # Path of annotation file
        img_prefix='data/coco/train2017/',  # Prefix of image path
        pipeline=train_pipeline),
    val=dict(  # Validation dataset config
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline),
    test=dict(  # Test dataset config, modify the ann_file for test-dev/test submission
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline)
)


###################################### METRICS ####################################
evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
    interval=1,  # Evaluation interval
    metric=['bbox'])  # Metrics used during evaluation

####################################### HYPER PARAMS #################################

optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',  # Type of optimizers, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13 for more details
    lr=0.02,  # Learning rate of optimizers, see detail usages of the parameters in the documentaion of PyTorch
    momentum=0.9,  # Momentum
    weight_decay=0.0001)  # Weight decay of SGD

optimizer_config = dict(  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
    grad_clip=None)  # Most of the methods do not use gradient clip

lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=0.015,  # The ratio of the starting learning rate used for warmup
    step=[15, 10])  # Steps to decay the learning rate

total_epochs = 25  # Total epochs to train the model

checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=1)  # The save interval is 1

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.

dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.

log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = 'work_dir'  # Directory to save the model checkpoints and logs for the current experiments.


_base_ = 'cascade_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco.py'
model = dict(
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
    roi_head = dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='BalancedL1Loss',
                    alpha=0.5,
                    gamma=1.5,
                    beta=1.0,
                    loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='BalancedL1Loss',
                alpha=0.5,
                gamma=1.5,
                beta=1.0,
                loss_weight=1.0))
        ]
    ),
    train_cfg=dict(
        rpn=dict(sampler=dict(neg_pos_ub=5), allowed_border=-1),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                pos_weight=-1,
                debug=False)
        ]
    )
)
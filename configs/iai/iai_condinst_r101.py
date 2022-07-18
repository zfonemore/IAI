# model settings
batch_size = 2
max_obj_num = 20
model = dict(
    type='IAICondInst',
    pretrained='torchvision://resnet101',
    id_cfg=dict(num_frames=5, batch_size=batch_size, max_obj_num=max_obj_num),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    lstt_block=dict(
        type='LSTTBlock',
        lstt_num = 1,
        max_obj_num=max_obj_num,
        in_channels=256,
        feat_channels=256,
        self_heads=4,
        attn_heads=2,
        global_mem_interval=3),
    bbox_head=dict(
        type='IAICondInstHead',
        num_classes=40,
        max_obj_num=max_obj_num,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_id=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1),
        center_sampling=True,
        center_sample_radius=1.5),
    # training and testing settings
    train_cfg = dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        id_score_thr=0.1,
        cls_score_thr=0.1,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=10))
# dataset settings
dataset_type = 'YTVOSDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.68, 116.779, 103.939], std=[58.393, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_id=True),
    dict(type='Resize', img_scale=[(649, 360), (960, 480)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_ids', 'scale']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_sub.json',
        img_prefix=data_root + 'train/JPEGImages',
        pipeline=train_pipeline,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_sub.json',
        img_prefix=data_root + 'valid/JPEGImages',
        pipeline=test_pipeline,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        with_track=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_sub.json',
        img_prefix=data_root + 'valid/JPEGImages',
        pipeline=test_pipeline,
        with_mask=False,
        with_label=False,
        test_mode=True,
        with_track=True))

runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
evaluation = dict(interval=1, metric=['bbox','segm'])
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './iai_condinst_r101/'
load_from = None
resume_from = None
workflow = [('train', 1)]
